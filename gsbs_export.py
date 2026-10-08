"""
Export a self-contained analysis package that replicates GUI results without
the GUI.

    <package>/
        README.md                how to run, what each step does
        requirements.txt         pinned versions
        analysis_config.json     every setting of the analysis
        run_all.py               runs the steps below in order
        toolkit/                 the exact (Qt-free) code the GUI used
        scripts/01_prepare_data.py   data download / NIfTI import
        scripts/02_fit_gsbs.py       GSBS fits
        scripts/03_analyses.py       summaries, overlap, clusters -> results/results.json
        scripts/04_figures.py        runs every figures/fig_*.py
        scripts/05_report.py         Methods/Results text and captions -> manuscript/
        figures/fig_<name>.py    one script per figure (+ .png/.pdf/.svg, _caption.md)
        manuscript/              methods/results (.md, .tex), figures.tex, references.bib
        results/results.json     every reported number
        data/                    small inputs (arrays, user transcripts or events)
        work/                    data and fits created by the scripts (not shared)
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT_MODULES = ("gsbs_core", "gsbs_nifti", "gsbs_atlas", "gsbs_surface", "gsbs_overlap",
                   "gsbs_clusters", "gsbs_stimulus", "gsbs_analysis", "gsbs_figures", "gsbs_report")


def software_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "nilearn", "nibabel", "scikit-learn", "matplotlib",
                "statesegmentation", "pandas"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def gui_commit() -> str | None:
    try:
        rev = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "*.py"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (rev + ("-modified" if dirty else "")) or None
    except Exception:                                              # noqa: BLE001
        return None


def example_key(source: str) -> str | None:
    """Which online example a session was imported from, if any."""
    import gsbs_nifti as nifti
    name = Path(source).name
    for ex in nifti.EXAMPLES:
        if ex.url and name == ex.url.rsplit("/", 1)[1]:
            return ex.key
    if "pixar123" in name:
        return "pixar"
    return None


# ---------------------------------------------------------------------------
# Script templates
# ---------------------------------------------------------------------------

COMMON = '''"""Shared paths and helpers for the analysis scripts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "toolkit"))
CONFIG = json.loads((ROOT / "analysis_config.json").read_text())
RESULTS = ROOT / "results" / "results.json"
WORK = ROOT / "work"


def session_dir() -> Path:
    """The imported session: work/session, or a folder named in work/session_path.txt."""
    link = WORK / "session_path.txt"
    return Path(link.read_text().strip()) if link.exists() else WORK / "session"


def session():
    import gsbs_nifti
    return gsbs_nifti.Session(session_dir())


def target_key(t):
    return t if t in ("whole_brain", "data") else int(t)


def load_target(t):
    """The fitted result for a target ('whole_brain', an ROI number, or 'data')."""
    import gsbs_core
    t = target_key(t)
    if t == "data":
        return gsbs_core.load_file(WORK / "result.npy").obj
    return session().load_result(t)


def load_results() -> dict:
    return json.loads(RESULTS.read_text())


def transcript_and_events():
    """The stimulus transcript and event onsets used by the analysis (or None)."""
    import numpy as np
    import gsbs_stimulus
    transcript = events = None
    if CONFIG["data"]["kind"] == "nifti":
        s = session()
        transcript, events = s.transcript(), s.event_times()
    if (ROOT / "data" / "transcript.tsv").exists():
        transcript = gsbs_stimulus.Transcript.load(ROOT / "data" / "transcript.tsv")
    if (ROOT / "data" / "events.txt").exists():
        events = np.loadtxt(ROOT / "data" / "events.txt", ndmin=1)
    return transcript, events
'''

PREPARE = '''"""Step 1: get the data and extract parcel and ROI voxel time series.

For an online example the NIfTI (and transcript) are downloaded once into
nilearn's data folder. For a local NIfTI, set data.path in analysis_config.json.
The import (Schaefer parcellation, cleaning, z-scoring, parcel averaging) uses
the settings in analysis_config.json under data.import.
"""

import shutil
import sys

from _common import CONFIG, ROOT, WORK, session_dir

import gsbs_nifti

data = CONFIG["data"]
if data["kind"] != "nifti":
    print("Array input: nothing to prepare (data/input.npy is used directly).")
    sys.exit(0)

out = session_dir()
if (out / "session.json").exists():
    print(f"Session already prepared: {out}")
    sys.exit(0)

transcript = None
confounds = None
if data.get("example"):
    files = gsbs_nifti.fetch_example(data["example"], progress=None)
    nifti_path, transcript = files["nifti"], files.get("transcript")
    if data["import"].get("confounds"):
        confounds = files["confounds"]
else:
    nifti_path = data["path"]
    confounds = data["import"].get("confounds") or None

imp = dict(data["import"])
imp["confounds"] = confounds
WORK.mkdir(exist_ok=True)


class Log:
    def put(self, msg):
        if msg[0] == "stage":
            print(" ", msg[1])
        elif msg[0] == "error":
            print(msg[1])
            sys.exit(1)


gsbs_nifti.run_import(Log(), gsbs_nifti.ImportParams(nifti=str(nifti_path), session_dir=str(out), **imp))
s = gsbs_nifti.Session(out)
if transcript is not None:
    s.attach_transcript(transcript)
print(f"Prepared {out}: {s.n_time} TRs, {s.spec.n_rois} parcels")
'''

FIT = '''"""Step 2: fit GSBS to every target listed in analysis_config.json (fits.targets).

Existing fits are reused when their parameters match the configuration; pass
--refit to fit everything again, and --workers N to fit N targets at a time.
fast_GSBS gives results identical to GSBS.
"""

import os
import sys
import time

from _common import CONFIG, ROOT, WORK, session, session_dir, target_key


def fit(x, params, engine_name):
    import statesegmentation
    obj = getattr(statesegmentation, engine_name)(x=x, **params)
    t0 = time.monotonic()
    obj.fit(showProgressBar=False)
    return obj, time.monotonic() - t0


def fit_target(t):
    """Fit one session target (run in a worker process)."""
    import gsbs_nifti
    s = gsbs_nifti.Session(session_dir())
    return (t, *fit(s.data(target_key(t)), CONFIG["fits"]["params"][str(t)], CONFIG["fits"]["engine"]))


def main():
    import gsbs_core
    fits = CONFIG["fits"]
    refit = "--refit" in sys.argv
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1

    if CONFIG["data"]["kind"] != "nifti":
        import numpy as np
        out = WORK / "result.npy"
        if out.exists() and not refit:
            print(f"Using existing fit {out}")
            return
        obj, secs = fit(np.load(ROOT / "data" / "input.npy"), fits["params"]["data"], fits["engine"])
        WORK.mkdir(exist_ok=True)
        gsbs_core.save_object(obj, out)
        print(f"data: best {gsbs_core.best_k(obj)} states ({secs:.0f} s)")
        return

    s = session()
    todo = []
    for t in fits["targets"]:
        existing = s.load_result(target_key(t))
        if existing is not None and not refit and gsbs_core.fit_params(existing) == fits["params"][str(t)]:
            print(f"{t}: using existing fit (best {gsbs_core.best_k(existing)} states)")
        else:
            todo.append(t)
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = "1"                    # one BLAS thread per worker
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(fit_target, todo)
            for t, obj, secs in results:
                obj.gsbs_gui_meta = s.target_meta(target_key(t))
                s.record(target_key(t), obj, secs, fits["engine"])
                print(f"{t}: best {gsbs_core.best_k(obj)} states ({secs:.0f} s)", flush=True)
    else:
        for t in todo:
            _, obj, secs = fit_target(t)
            obj.gsbs_gui_meta = s.target_meta(target_key(t))
            s.record(target_key(t), obj, secs, fits["engine"])
            print(f"{t}: best {gsbs_core.best_k(obj)} states ({secs:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
'''

ANALYSES = '''"""Step 3: compute every reported number and save results/results.json."""

import json

import numpy as np

from _common import CONFIG, RESULTS, load_target, session, target_key, transcript_and_events

import gsbs_analysis as A
import gsbs_atlas
import gsbs_core

cfg = CONFIG
seg_cfg = cfg["segmentation"]
obj = load_target(seg_cfg["target"])
tr = cfg["data"].get("tr")
res = {"segmentation": A.segmentation_summary(obj, seg_cfg["k"], tr)}
k = seg_cfg["k"]
edges = gsbs_core.state_edges(obj.all_bounds[k], obj.x.shape[0])

spec = gsbs_atlas.Schaefer.from_meta(cfg["data"].get("atlas"))
info = gsbs_atlas.parcel_info(spec) if spec else None

# Strongest boundary of the shown whole-brain solution
if cfg["data"]["kind"] == "nifti" and target_key(seg_cfg["target"]) == "whole_brain" and k > 1:
    patterns, _ = A.state_patterns(obj, k)
    strengths = np.array(res["segmentation"]["strengths"])
    b = int(np.argmax(strengths)) + 1                     # boundary b separates states b and b + 1
    change = patterns[b] - patterns[b - 1]
    nets = {n: float(np.mean(change[info.networks == n])) for n in info.network_order}
    res["boundary_example"] = {"index": b, "tr": int(edges[b]), "strength": float(strengths[b - 1]),
                               "top_network": max(nets, key=lambda n: abs(nets[n])),
                               "network_change": nets}

# Region-level timescales
if cfg["data"]["kind"] == "nifti" and any(t != "whole_brain" for t in cfg["fits"]["targets"]):
    res["timescales"] = A.roi_timescales(session())

# Stimulus boundaries and their overlap with the shown segmentation
st = cfg.get("stimulus")
if st and tr:
    transcript, events = transcript_and_events()
    times = A.stimulus_times(transcript, events, st["boundaries"], st["pause_s"])
    stim_trs = A.stimulus_trs(times, tr, obj.x.shape[0], st["delay_s"])
    res["stimulus_trs"] = stim_trs.tolist()
    res["stimulus_info"] = A.stimulus_info(transcript, events, st["boundaries"], st["pause_s"], tr,
                                           obj.x.shape[0], st["delay_s"])
    res["stimulus_overlap"] = A.pair_overlap(stim_trs, edges[1:-1], obj.x.shape[0],
                                             st.get("tolerance", 0), A.N_PERM, seed=0)

# Per-ROI overlap with a reference (stimulus at one or several delays, or the whole brain)
om = cfg.get("overlap_map")
if om:
    s = session()
    ref = om["reference"]
    if ref["kind"] == "stimulus":
        transcript, events = transcript_and_events()
        times = A.stimulus_times(transcript, events, ref["boundaries"], ref["pause_s"])
        refs = [A.stimulus_trs(times, s.tr, s.n_time, d) for d in om["delays"]]
        delays = om["delays"]
    else:
        refs, delays = [s.best_bounds(target_key(ref["target"]))], None
    r = A.roi_overlap_map(s, refs, delays, om.get("n_perm", A.N_PERM))
    sig = {roi: q < om.get("alpha", 0.05) for roi, q in zip(r["rois"], r["q"])}
    r["network_means"] = A.network_means(dict(zip(r["rois"], r["or"])), info, sig)
    if r["grid"]:
        r["network_delay_means"] = A.network_means(dict(zip(r["rois"], r["best_delay"])), info, sig)
    res["overlap_map"] = r

# Clusters of state patterns
cl = cfg.get("clusters")
if cl:
    res["clusters"] = A.state_clusters(load_target(cl["target"]), cl["k"], cl["method"], cl["params"])

RESULTS.parent.mkdir(exist_ok=True)
RESULTS.write_text(json.dumps(res, indent=1, default=float))
print(f"Saved {RESULTS}")
'''

FIGURES = '''"""Step 4: draw every figure by running its own script in figures/."""

import runpy
import sys

from _common import ROOT

for script in sorted((ROOT / "figures").glob("fig_*.py")):
    print(" ", script.name)
    runpy.run_path(str(script), run_name="__main__")
'''

REPORT = '''"""Step 5: write the Methods and Results text and the figure captions."""

from _common import CONFIG, ROOT, load_results

import gsbs_report

res = load_results()
out = gsbs_report.write_all(CONFIG, res, ROOT / "manuscript")
for i, fid in enumerate(out["figures"], 1):
    (ROOT / "figures" / f"fig_{fid}_caption.md").write_text(f"**Figure {i}.** {out['captions'][fid]}\\n")
print("Wrote manuscript/methods.md, results.md, .tex versions, figures.tex and references.bib")
'''

FIG_HEADER = '''"""{title}

Drawn from results/results.json and the session data by gsbs_figures.{func}().
Run from anywhere: python figures/{name}.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _common import CONFIG, ROOT, load_results, load_target  # noqa: E402

import numpy as np  # noqa: E402
import gsbs_atlas  # noqa: E402
import gsbs_core  # noqa: E402
import gsbs_figures  # noqa: E402

res = load_results()
tr = CONFIG["data"].get("tr")
spec = gsbs_atlas.Schaefer.from_meta(CONFIG["data"].get("atlas"))
info = gsbs_atlas.parcel_info(spec) if spec else None
OUT = str(ROOT / "figures" / "{fid}")


def surface():
    import gsbs_surface
    return gsbs_surface.SurfaceStrip.load(height=420)

'''

FIG_BODIES = {
    "segmentation": ("Figure: neural state segmentation (t-distance, correlation matrix, timeseries).", "segmentation", '''
obj = load_target(CONFIG["segmentation"]["target"])
seg = res["segmentation"]
k = seg["k"]
ks = gsbs_core.solution_ks(obj)
edges = gsbs_core.state_edges(obj.all_bounds[k], obj.x.shape[0])
columns = "parcels" if CONFIG["segmentation"]["target"] == "whole_brain" else CONFIG["data"].get("columns", "voxels")
fig = gsbs_figures.segmentation(obj.x, ks, obj.tdists[ks], seg["best_k"], k, edges, tr,
                                res.get("stimulus_trs"), columns=columns)
gsbs_figures.save(fig, OUT)
'''),
    "boundary": ("Figure: activity patterns around the strongest state boundary.", "boundary", '''
obj = load_target("whole_brain")
k = res["segmentation"]["k"]
b = res["boundary_example"]["index"]
from gsbs_analysis import state_patterns  # noqa: E402
patterns, _ = state_patterns(obj, k)
before, after = patterns[b - 1], patterns[b]
lim = float(np.percentile(np.abs(patterns), 98))
clim = float(np.percentile(np.abs(np.diff(patterns, axis=0)), 98))
fig = gsbs_figures.boundary(surface(), spec, info, before, after, after - before, lim, clim,
                            labels=(f"State {b}", f"State {b + 1}", "Change"))
gsbs_figures.save(fig, OUT)
'''),
    "timescales": ("Figure: optimal number of states per parcel.", "roi_map", '''
ts = res["timescales"]
vals = np.full(spec.n_rois, np.nan)
vals[np.array(ts["rois"]) - 1] = ts["nstates"]
lo, hi = float(np.min(ts["nstates"])), float(np.max(ts["nstates"]))
nets = {n: float(np.mean([v for r, v in zip(ts["rois"], ts["nstates"]) if info.networks[r - 1] == n]))
        for n in info.network_order if any(info.networks[r - 1] == n for r in ts["rois"])}
fig = gsbs_figures.roi_map(surface(), spec, info, vals, "viridis", lo, hi, "optimal number of states",
                           net_values=nets, sequential=True)
gsbs_figures.save(fig, OUT)
'''),
    "stimulus_overlap": ("Figure: overlap between neural state boundaries and stimulus boundaries.", "overlap_pair", '''
so = res["stimulus_overlap"]
st = CONFIG["stimulus"]
fig = gsbs_figures.overlap_pair(res["stimulus_trs"], res["segmentation"]["boundaries"], so["n_time"],
                                so["null"], so["observed"], so["expected"], st["label"],
                                "Neural states", tr=tr, tolerance=so["tolerance"])
gsbs_figures.save(fig, OUT)
'''),
    "overlap_map": ("Figure: per-parcel boundary overlap (FDR-thresholded) and best HRF delay.", "overlap_map", '''
om = res["overlap_map"]
alpha = CONFIG["overlap_map"].get("alpha", 0.05)
rois = np.array(om["rois"])
q = np.array(om["q"])
orv = np.full(spec.n_rois, np.nan)
orv[rois[q < alpha] - 1] = np.array(om["or"])[q < alpha]
lim = float(np.percentile(np.abs(om["or"]), 98)) or 1.0
counts = {n: f"{v['n_significant']}/{v['n']}" for n, v in om["network_means"].items()}
net_or = {n: v["mean"] for n, v in om["network_means"].items()}
delay_vals = net_delay = None
if om["grid"]:
    delay_vals = np.full(spec.n_rois, np.nan)
    delay_vals[rois - 1] = om["best_delay"]
    net_delay = {n: v["mean"] for n, v in om["network_delay_means"].items()}
fig = gsbs_figures.overlap_map(surface(), spec, info, orv, lim, delay_vals, om["delays"],
                               net_or, net_delay, counts)
gsbs_figures.save(fig, OUT)
'''),
    "clusters": ("Figure: clusters of state activity patterns, their timeline and transitions.", "clusters", '''
cl = res["clusters"]
use_surface = spec is not None and CONFIG["clusters"]["target"] == "whole_brain"
fig = gsbs_figures.clusters(cl["centroids"], cl["labels"], cl["edges"], cl["transitions"],
                            cl["time_fraction"], tr, surface() if use_surface else None,
                            spec if use_surface else None, info)
gsbs_figures.save(fig, OUT)
'''),
}

RUN_ALL = '''#!/usr/bin/env python3
"""Run the whole analysis: prepare data, fit GSBS, compute results, draw figures, write text.

Options
    --workers N      fit N targets at a time
    --skip-prepare   assume the session already exists
    --skip-fit       use existing fits only
    --refit          fit every target again even if fits exist
    --session PATH   use an existing session folder instead of work/session
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
args = sys.argv[1:]
if "--session" in args:
    path = args[args.index("--session") + 1]
    (ROOT / "work").mkdir(exist_ok=True)
    (ROOT / "work" / "session_path.txt").write_text(str(Path(path).resolve()))

steps = ["01_prepare_data.py", "02_fit_gsbs.py", "03_analyses.py", "04_figures.py", "05_report.py"]
for step in steps:
    if step.startswith("01") and "--skip-prepare" in args:
        continue
    if step.startswith("02") and "--skip-fit" in args:
        continue
    print(f"== {step}", flush=True)
    extra = []
    if step.startswith("02"):
        extra = [a for a in ("--refit",) if a in args]
        if "--workers" in args:
            extra += ["--workers", args[args.index("--workers") + 1]]
    code = subprocess.call([sys.executable, str(ROOT / "scripts" / step)] + extra)
    if code:
        sys.exit(code)
print("== done: results/results.json, figures/, manuscript/")
'''


# ---------------------------------------------------------------------------
# Writing a package
# ---------------------------------------------------------------------------

def figure_ids(cfg: dict) -> list[str]:
    """Figures this configuration can produce (the report drops any without results)."""
    ids = ["segmentation"]
    if cfg["data"]["kind"] == "nifti" and cfg["segmentation"]["target"] == "whole_brain":
        ids.append("boundary")
    if cfg["data"]["kind"] == "nifti" and sum(t != "whole_brain" for t in cfg["fits"]["targets"]) >= 2:
        ids.append("timescales")
    if cfg.get("stimulus"):
        ids.append("stimulus_overlap")
    if cfg.get("overlap_map"):
        ids.append("overlap_map")
    if cfg.get("clusters"):
        ids.append("clusters")
    return ids


def readme(cfg: dict) -> str:
    data = cfg["data"]
    src = (f"the online example `{data['example']}` (downloaded automatically)" if data.get("example")
           else f"`{data.get('path', 'data/input.npy')}`")
    return f"""# GSBS analysis package

Generated by GSBS GUI on {cfg['package']['created']}. Running this package without the GUI
reproduces every number in `results/results.json`, every figure in `figures/` and the
Methods and Results text in `manuscript/`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --workers 4     # fits 4 targets at a time
```

The data are {src}.{" To use another copy of the NIfTI, set `data.path` in `analysis_config.json`." if data["kind"] == "nifti" and not data.get("example") else ""}
Fitting is the slow step (see `analysis_config.json` for the targets); everything else takes
seconds to a few minutes.

## Steps

| Script | Does |
|---|---|
| `scripts/01_prepare_data.py` | gets the data and extracts parcel and ROI voxel time series |
| `scripts/02_fit_gsbs.py` | fits GSBS to every target (reuses matching fits; `--refit` to redo) |
| `scripts/03_analyses.py` | segmentation summaries, overlap, clusters: `results/results.json` |
| `scripts/04_figures.py` | runs every `figures/fig_*.py` (PNG, PDF and SVG) |
| `scripts/05_report.py` | `manuscript/` Methods and Results (Markdown and LaTeX), captions, references |

`toolkit/` is a copy of the code GSBS GUI used for these analyses, so the scripts compute
exactly what the GUI showed. `analysis_config.json` holds every setting; edit it and rerun
to change the analysis. Items the generator could not know are marked `[TODO: ...]` in
the manuscript text.
"""


def write_package(cfg: dict, out: str | Path, gui_result=None, transcript=None, events=None,
                  array=None) -> Path:
    """Write the package files (no analysis is run here)."""
    out = Path(out)
    for d in ("toolkit", "scripts", "figures", "manuscript", "results", "data", "work"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for mod in TOOLKIT_MODULES:
        shutil.copy(HERE / f"{mod}.py", out / "toolkit" / f"{mod}.py")
    for name, text in (("_common.py", COMMON), ("01_prepare_data.py", PREPARE), ("02_fit_gsbs.py", FIT),
                       ("03_analyses.py", ANALYSES), ("04_figures.py", FIGURES), ("05_report.py", REPORT)):
        (out / "scripts" / name).write_text(text)
    for old in (out / "figures").glob("fig_*.py"):
        old.unlink()
    for fid in figure_ids(cfg):
        title, func, body = FIG_BODIES[fid]
        (out / "figures" / f"fig_{fid}.py").write_text(
            FIG_HEADER.format(title=title, func=func, name=f"fig_{fid}", fid=fid) + body.lstrip("\n"))
    (out / "run_all.py").write_text(RUN_ALL)
    (out / "analysis_config.json").write_text(json.dumps(cfg, indent=2))
    sw = cfg["package"]["software"]
    req = [f"{p}=={sw[p]}" for p in ("numpy", "scipy", "nilearn", "nibabel", "scikit-learn",
                                     "matplotlib", "pandas") if p in sw]
    req.append("statesegmentation @ git+https://github.com/drgzkr/statesegmentation.git"
               f"  # version {sw.get('statesegmentation', '?')} was used")
    (out / "requirements.txt").write_text("\n".join(req) + "\n")
    (out / "README.md").write_text(readme(cfg))
    (out / ".gitignore").write_text("work/\n__pycache__/\n")
    if array is not None:
        import numpy as np
        np.save(out / "data" / "input.npy", array)
    if gui_result is not None:
        import gsbs_core
        gsbs_core.save_object(gui_result, out / "work" / "result.npy")
    if transcript is not None:
        transcript.save(out / "data" / "transcript.tsv")
    if events is not None:
        import numpy as np
        np.savetxt(out / "data" / "events.txt", np.asarray(events, float), fmt="%.3f")
    return out


def run_package(out: str | Path, args: list[str], log=print) -> int:
    """Run run_all.py in a separate Python process, streaming its output to `log`."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    proc = subprocess.Popen([sys.executable, str(Path(out) / "run_all.py"), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        log(line.rstrip())
    return proc.wait()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")
