"""
NIfTI import and analysis sessions.

Importing a 4D NIfTI creates a session folder:

    <name>_gsbs/
        session.json        source, atlas, TR, preprocessing, parcel voxel counts
        parcels.npy         (timepoints x parcels) whole-brain parcel timeseries
        voxels.npy          (voxels x timepoints) float32, every atlas voxel, sorted by
                            parcel so one ROI's voxels are a contiguous block on disk
        voxel_labels.npy    parcel id of each row of voxels.npy
        voxel_ijk.npy       voxel indices in the source image
        results/            fitted GSBS objects: whole_brain.npy, roi_0001.npy, ...
        results/summary.json  one line of statistics per fitted target

Preprocessing follows the GSBS scripts (run_gsbs_sherlock.py and friends):
optional nuisance cleaning, z-score every voxel over time, then average voxels
within each parcel. Parcel timeseries can be z-scored again (neuro_tools'
compute_roi_averaged_matrix does this).
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import gsbs_atlas as atlas

WHOLE_BRAIN = "whole_brain"
DEFAULT_CONFOUNDS = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
                     "white_matter", "csf")


# ---------------------------------------------------------------------------
# Online example data
# ---------------------------------------------------------------------------
#
# pixar      nilearn's development fMRI sample: one adult watching Pixar's "Partly
#            Cloudy" (Richardson et al., 2018, Nat Commun 9:1027). fMRIPrep, MNI, 4 mm,
#            168 volumes. The header records TR = 1 without units; the TR is 2 s.
# narratives Narratives (Nastase et al., 2021, Sci Data 8:250; CC0), one participant per
#            story, from the INDI S3 mirror over plain HTTPS. These are the afni-nosmooth
#            "desc-clean" images: MNI152NLin2009cAsym, 3 x 3 x 4 mm, TR 1.5 s, with six
#            motion parameters, five CSF/white-matter components, a 128 s cosine high-pass
#            and 1st/2nd-order polynomials already regressed out, so no confounds are
#            applied again. Each scan has a few more volumes than the story itself.
#            The story's transcript comes with it: Gentle word alignments (times relative
#            to the audio file) placed on the scan with the onset of the audio file from
#            the participant's events.tsv. Stimulus: CC0 for Sherlock and The 21st Year;
#            Tunnel is shared by Narratives for non-commercial research (fair use).

_NARRATIVES_ROOT = "https://fcp-indi.s3.amazonaws.com/data/Projects/narratives/"
_NARRATIVES = ("https://fcp-indi.s3.amazonaws.com/data/Projects/narratives/derivatives/"
               "afni-nosmooth/{sub}/func/{sub}_task-{task}_space-MNI152NLin2009cAsym_"
               "res-native_desc-clean_bold.nii.gz")


@dataclass(frozen=True)
class Example:
    key: str
    title: str
    description: str
    tr: float
    volumes: int
    size_mb: float
    url: str | None = None          # None: fetched through nilearn
    tr_note: str = ""
    sub: str = ""                   # Narratives participant and story, for the transcript
    task: str = ""


EXAMPLES = (
    Example("pixar", "Movie, 5.6 min: Pixar \u201cPartly Cloudy\u201d (nilearn)",
            "nilearn development fMRI sample: one adult watching Pixar's \u201cPartly Cloudy\u201d "
            "(Richardson et al., 2018), fMRIPrep-preprocessed, MNI space, 4 mm, 168 volumes "
            "at TR 2 s.",
            2.0, 168, 6,
            tr_note="Set from the dataset description (the header says 1 without units)."),
    Example("narratives-sherlock",
            "Spoken story, 18 min: recall of \u201cSherlock\u201d (Narratives)",
            "Narratives (Nastase et al., 2021), participant sub-151 listening to an 18-minute "
            "spoken recall of the first episode of BBC\u2019s Sherlock. Confound-cleaned, MNI "
            "space, 3\u00d73\u00d74 mm, 724 volumes at TR 1.5 s.",
            1.5, 724, 138, _NARRATIVES.format(sub="sub-151", task="sherlock"),
            sub="sub-151", task="sherlock"),
    Example("narratives-tunnel",
            "Spoken story, 26 min: \u201cTunnel Under the World\u201d (Narratives)",
            "Narratives (Nastase et al., 2021), participant sub-041 listening to Frederik "
            "Pohl\u2019s science-fiction story as broadcast by the X Minus One radio series. "
            "Confound-cleaned, MNI space, 3\u00d73\u00d74 mm, 1040 volumes at TR 1.5 s.",
            1.5, 1040, 199, _NARRATIVES.format(sub="sub-041", task="tunnel"),
            sub="sub-041", task="tunnel"),
    Example("narratives-21styear",
            "Spoken story, 56 min: \u201cThe 21st Year\u201d (Narratives)",
            "Narratives (Nastase et al., 2021), participant sub-257 listening to Christina "
            "Lazaridi\u2019s \u201cThe 21st Year\u201d, two storylines that converge. "
            "Confound-cleaned, MNI space, 3\u00d73\u00d74 mm, 2249 volumes at TR 1.5 s. "
            "Loading it needs about 3.3 GB of memory, and fits take longer.",
            1.5, 2249, 430, _NARRATIVES.format(sub="sub-257", task="21styear"),
            sub="sub-257", task="21styear"),
)
EXAMPLE_BY_KEY = {e.key: e for e in EXAMPLES}


class DownloadCancelled(Exception):
    pass


def example_cache_dir() -> Path:
    from nilearn.datasets import get_data_dirs
    return Path(get_data_dirs()[0]) / "gsbs_gui_examples"


def fetch_example(key: str = "pixar", progress=None) -> dict:
    """Download an example once and return {"nifti", "confounds", "tr"}.

    `progress(done_bytes, total_bytes)` is called during downloads; raising
    DownloadCancelled from it stops the download and removes the partial file.
    """
    ex = EXAMPLE_BY_KEY[key]
    if ex.url is None:
        from nilearn import datasets
        for group in ("adult", "adults"):  # the accepted spelling differs between nilearn versions
            try:
                d = datasets.fetch_development_fmri(n_subjects=1, age_group=group, verbose=0)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("nilearn rejected both spellings of the adult age group.")
        return {"nifti": d.func[0], "confounds": d.confounds[0], "tr": ex.tr, "transcript": None}

    import urllib.request
    path = example_cache_dir() / ex.url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        try:
            with urllib.request.urlopen(ex.url, timeout=60) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
            if total and done != total:
                raise OSError(f"Download incomplete: {done} of {total} bytes.")
            part.rename(path)
        finally:
            part.unlink(missing_ok=True)
    return {"nifti": str(path), "confounds": None, "tr": ex.tr,
            "transcript": _fetch_narratives_transcript(ex)}


def _fetch_narratives_transcript(ex: Example):
    import urllib.request

    from gsbs_stimulus import Transcript, narratives_audio_onset
    cache = example_cache_dir()
    align = cache / f"{ex.task}_align.json"
    events = cache / f"{ex.sub}_task-{ex.task}_events.tsv"
    for local, rel in ((align, f"stimuli/gentle/{ex.task}/align.json"),
                       (events, f"{ex.sub}/func/{ex.sub}_task-{ex.task}_events.tsv")):
        if not local.exists():
            part = local.with_name(local.name + ".part")
            urllib.request.urlretrieve(_NARRATIVES_ROOT + rel, part)
            part.rename(local)
    onset = narratives_audio_onset(events, f"{ex.task}_audio.wav")
    return Transcript.from_gentle(align, onset, source=f"Narratives \u201c{ex.task}\u201d transcript, "
                                  f"Gentle word alignment, audio onset {onset:g} s ({ex.sub})")


# ---------------------------------------------------------------------------
# Ready-made demo session (Narratives "Sherlock", fitted)
# ---------------------------------------------------------------------------
#
# data/demo_sherlock holds the fitted results (whole brain and 12 ROIs), their
# summary, the transcript and the import settings (tools/make_sherlock_demo.py).
# Installing re-imports the example NIfTI with those settings, checks that the
# parcel data equal the stored whole-brain fit's data, then adds the results.

READY_DEMO_DIR = Path(__file__).resolve().parent / "data" / "demo_sherlock"
READY_DEMO = Example(
    "ready-sherlock",
    "Ready-made session: \u201cSherlock\u201d recall, whole brain + 12 ROIs fitted",
    "The 18-minute Narratives \u201cSherlock\u201d recall (sub-151) imported as Schaefer 400/17, "
    "with GSBS already fitted for the whole brain and 12 ROIs (visual, auditory, default "
    "B temporal, temporo-parietal; one is flagged for having its best solution at kmax) and "
    "the transcript attached. Downloads the NIfTI once (\u2248 138 MB), then imports it "
    "(a few seconds).",
    1.5, 724, 138)


class _Collect:
    def __init__(self, stage=None):
        self.items, self.stage = [], stage

    def put(self, msg):
        self.items.append(msg)
        if self.stage and msg[0] == "stage":
            self.stage(msg[1])


def install_demo_session(progress=None, stage=None) -> Path:
    """Build (once) and return the ready-made demo session folder."""
    import gzip
    import shutil

    manifest = json.loads((READY_DEMO_DIR / "manifest.json").read_text())
    out = example_cache_dir() / "sherlock_demo_gsbs"
    if (out / "session.json").exists() and (out / "results" / "whole_brain.npy").exists():
        return out
    files = fetch_example(manifest["example"], progress=progress)
    if stage:
        stage("Importing")
    q = _Collect(stage)
    run_import(q, ImportParams(nifti=files["nifti"], session_dir=str(out), **manifest["import"]))
    if q.items[-1][0] != "done":
        raise RuntimeError(q.items[-1][1])
    session = Session(out)
    copied = []
    for gz in sorted((READY_DEMO_DIR / "results").glob("*.npy.gz")):
        dst = out / "results" / gz.name[:-3]
        with gzip.open(gz, "rb") as src, open(dst, "wb") as f:
            shutil.copyfileobj(src, f)
        copied.append(dst)
    shutil.copy(READY_DEMO_DIR / "results" / "summary.json", out / "results" / "summary.json")
    copied.append(out / "results" / "summary.json")
    wb = session.load_result(WHOLE_BRAIN)
    if wb is None or wb.x.shape != session.parcels().shape or not np.allclose(
            wb.x, session.parcels(), atol=1e-5):
        for f in copied:
            f.unlink(missing_ok=True)
        raise RuntimeError("The re-imported data differ from the stored fits (a different nilearn "
                           f"version?). The imported session is at {out}; fit it there instead.")
    session.stimulus_dir.mkdir(exist_ok=True)
    for f in ("transcript.tsv", "transcript.json"):
        shutil.copy(READY_DEMO_DIR / f, session.stimulus_dir / f)
    return out


def nifti_stem(path: str | os.PathLike) -> str:
    name = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(path).stem


def default_session_dir(nifti: str | os.PathLike) -> Path:
    return Path(nifti).resolve().parent / f"{nifti_stem(nifti)}_gsbs"


def peek_nifti(path: str | os.PathLike) -> dict:
    """Shape, TR and in-memory size of a NIfTI, read from the header only."""
    import nibabel as nib
    img = nib.load(str(path))
    shape = img.shape
    if len(shape) != 4:
        raise ValueError(f"Expected a 4D NIfTI, got shape {shape}.")
    zooms = img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 else 0.0
    units = img.header.get_xyzt_units()[1]
    note = ""
    if units == "msec":
        tr /= 1000.0
    elif units == "usec":
        tr /= 1e6
    if units == "sec" and tr > 100:      # e.g. 1500 labelled as seconds: written in ms
        note = f"The header says {tr:g} s; read as {tr / 1000:g} s (milliseconds). Check this TR."
        tr /= 1000.0
    elif units not in ("sec", "msec", "usec"):
        note = "The header does not record time units: check this TR."
    return {"shape": tuple(int(s) for s in shape), "tr": tr or None,
            "tr_units_known": units in ("sec", "msec", "usec"), "tr_note": note,
            "voxel_mm": tuple(float(z) for z in zooms[:3]),
            "bytes": int(np.prod(shape)) * 4}


@dataclass
class ImportParams:
    nifti: str
    session_dir: str
    n_rois: int = 400
    yeo_networks: int = 17
    tr: float | None = None
    confounds: str | None = None        # fMRIPrep-style .tsv
    detrend: bool = False
    high_pass: float | None = None      # Hz
    zscore_parcels: bool = False


# ---------------------------------------------------------------------------
# Import (runs in a child process)
# ---------------------------------------------------------------------------
#
# Messages: ("stage", text, fraction), ("done", session_dir), ("error", traceback)

def _zscore_rows(a: np.ndarray) -> np.ndarray:
    """Z-score each row over its columns (population SD, as scipy.stats.zscore)."""
    a = a - a.mean(axis=1, keepdims=True)
    sd = a.std(axis=1, keepdims=True)
    return np.divide(a, sd, out=np.zeros_like(a), where=sd > 0)


def _load_confounds(path: str, n_time: int) -> tuple[np.ndarray, list]:
    """Confound regressors and the names of the columns used."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t")
    cols = [c for c in DEFAULT_CONFOUNDS if c in df.columns]
    if not cols:
        cols = list(df.select_dtypes("number").columns)
    if len(df) != n_time:
        raise ValueError(f"Confounds file has {len(df)} rows; the NIfTI has {n_time} volumes.")
    return df[cols].fillna(0).to_numpy(float), cols


def run_import(queue, params: ImportParams) -> None:
    try:
        _import(queue, params)
    except BaseException:                                   # noqa: BLE001
        queue.put(("error", traceback.format_exc()))


def _import(queue, p: ImportParams) -> None:
    import nibabel as nib
    from nilearn import image, signal

    def stage(text, frac):
        queue.put(("stage", text, frac))

    out = Path(p.session_dir)
    info = peek_nifti(p.nifti)
    n_time = info["shape"][3]
    tr = p.tr or info["tr"]
    spec = atlas.Schaefer(p.n_rois, p.yeo_networks)

    stage("Preparing the Schaefer atlas", 0.02)
    resolution = 1 if min(info["voxel_mm"]) < 2 else 2
    img = nib.load(p.nifti)
    ref = image.index_img(img, 0)
    atlas_img = image.resample_to_img(atlas.fetch_volume(spec, resolution), ref,
                                      interpolation="nearest", force_resample=True,
                                      copy_header=True)
    labels_vol = np.asarray(atlas_img.dataobj).astype(np.int32)
    mask = labels_vol > 0
    order = np.argsort(labels_vol[mask], kind="stable")
    labels = labels_vol[mask][order]
    ijk = np.argwhere(mask)[order]

    stage(f"Reading {n_time} volumes", 0.08)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    voxels = data[mask][order]                              # (V, T)
    del data

    conf_cols = []
    if p.confounds or p.detrend or p.high_pass:
        stage("Cleaning (confounds / detrend / high-pass)", 0.45)
        conf, conf_cols = _load_confounds(p.confounds, n_time) if p.confounds else (None, [])
        if p.high_pass and not tr:
            raise ValueError("High-pass filtering needs the TR.")
        for a in range(0, len(voxels), 20000):
            block = voxels[a:a + 20000].T.astype(np.float64)
            voxels[a:a + 20000] = signal.clean(
                block, detrend=p.detrend, high_pass=p.high_pass, t_r=tr,
                confounds=conf, standardize=None).T

    stage("Z-scoring voxels", 0.65)
    keep = voxels.std(axis=1) > 0                           # drop empty / constant voxels
    voxels, labels, ijk = voxels[keep], labels[keep], ijk[keep]
    for a in range(0, len(voxels), 20000):
        voxels[a:a + 20000] = _zscore_rows(voxels[a:a + 20000].astype(np.float64))

    stage("Averaging within parcels", 0.8)
    counts = np.bincount(labels, minlength=spec.n_rois + 1)[1:]
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    parcels = np.zeros((spec.n_rois, n_time))
    present = counts > 0
    sums = np.add.reduceat(voxels.astype(np.float64), starts[present], axis=0)
    parcels[present] = sums / counts[present, None]
    if p.zscore_parcels:
        parcels[present] = _zscore_rows(parcels[present])

    stage("Writing the session", 0.9)
    (out / "results").mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out / "voxels.npy", mode="w+", dtype=np.float32,
                                   shape=voxels.shape)
    mm[:] = voxels
    mm.flush()
    del mm
    np.save(out / "voxel_labels.npy", labels)
    np.save(out / "voxel_ijk.npy", ijk.astype(np.int16))
    np.save(out / "parcels.npy", parcels.T)
    meta = {
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(Path(p.nifti).resolve()),
        "shape": info["shape"],
        "voxel_mm": info["voxel_mm"],
        "tr": tr,
        "atlas": spec.to_meta() | {"resolution_mm": resolution},
        "preprocessing": {
            "confounds": p.confounds, "detrend": p.detrend, "high_pass": p.high_pass,
            "voxel_zscore": True, "zscore_parcels": p.zscore_parcels,
            "confound_columns": conf_cols if (p.confounds or p.detrend or p.high_pass) else [],
        },
        "voxels_per_parcel": counts.tolist(),
        "missing_parcels": (np.flatnonzero(~present) + 1).tolist(),
    }
    (out / "session.json").write_text(json.dumps(meta, indent=2))
    stage("Done", 1.0)
    queue.put(("done", str(out)))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class TargetSummary:
    nstates: int
    n_time: int
    fit_seconds: float
    engine: str
    params: dict = field(default_factory=dict)
    median_duration: float = 0.0        # TRs
    bounds: list = field(default_factory=list)   # boundary TRs of the best solution
    at_limit: bool = False          # best k is the largest searched (no optimum found)

    def to_json(self) -> dict:
        return asdict(self)


class Session:
    """An imported NIfTI: parcel data, ROI voxel data and fitted results."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        meta_path = self.path / "session.json"
        if not meta_path.exists():
            raise ValueError(f"{self.path} is not a GSBS session (no session.json).")
        self.meta = json.loads(meta_path.read_text())
        self.spec = atlas.Schaefer.from_meta(self.meta["atlas"])
        self.tr = self.meta.get("tr")
        self.n_time = int(self.meta["shape"][3])
        counts = np.asarray(self.meta["voxels_per_parcel"])
        self.voxel_counts = counts
        self._starts = np.r_[0, np.cumsum(counts)[:-1]]
        self._voxels = None
        self._parcels = None
        self._summary = None

    @property
    def name(self) -> str:
        return self.path.name

    # ---- data ---------------------------------------------------------------

    def parcels(self) -> np.ndarray:
        if self._parcels is None:
            self._parcels = np.load(self.path / "parcels.npy")
        return self._parcels

    def roi_voxels(self, roi: int) -> np.ndarray:
        """(timepoints x voxels) z-scored voxel data of parcel `roi` (1-based)."""
        if self._voxels is None:
            self._voxels = np.load(self.path / "voxels.npy", mmap_mode="r")
        a = int(self._starts[roi - 1])
        return np.asarray(self._voxels[a:a + int(self.voxel_counts[roi - 1])],
                          dtype=np.float64).T

    def data(self, target) -> np.ndarray:
        return self.parcels() if target == WHOLE_BRAIN else self.roi_voxels(int(target))

    def target_meta(self, target) -> dict:
        meta = {"session": str(self.path), "tr": self.tr, **self.spec.to_meta()}
        if target == WHOLE_BRAIN:
            meta["kind"] = "parcels"
        else:
            meta |= {"kind": "roi_voxels", "roi": int(target)}
        return meta

    def target_label(self, target) -> str:
        if target == WHOLE_BRAIN:
            return f"Whole brain ({self.spec.n_rois} parcels)"
        info = atlas.parcel_info(self.spec)
        return f"ROI {int(target)}: {atlas.short_name(info.names[int(target) - 1])}"

    # ---- results ------------------------------------------------------------

    def result_path(self, target) -> Path:
        name = WHOLE_BRAIN if target == WHOLE_BRAIN else f"roi_{int(target):04d}"
        return self.path / "results" / f"{name}.npy"

    def load_result(self, target):
        path = self.result_path(target)
        if not path.exists():
            return None
        import gsbs_core as core
        return core.load_file(path).obj

    def summary(self) -> dict:
        """{target: TargetSummary}; target is WHOLE_BRAIN or an int ROI id."""
        if self._summary is None:
            path = self.path / "results" / "summary.json"
            raw = json.loads(path.read_text()) if path.exists() else {}
            fields = set(TargetSummary.__dataclass_fields__)
            self._summary = {(k if k == WHOLE_BRAIN else int(k)):
                             TargetSummary(**{f: x for f, x in v.items() if f in fields})
                             for k, v in raw.items()}
        return self._summary

    def record(self, target, obj, fit_seconds: float, engine: str) -> TargetSummary:
        """Save a fitted result and its summary line (called by the GUI process only)."""
        import gsbs_core as core
        if not hasattr(obj, "gsbs_gui_meta"):
            obj.gsbs_gui_meta = self.target_meta(target)
        path = self.result_path(target)
        tmp = path.with_name(path.stem + ".tmp.npy")
        core.save_object(obj, tmp)
        os.replace(tmp, path)
        return self.add_summary(target, summarise(obj, fit_seconds, engine))

    def best_bounds(self, target) -> np.ndarray | None:
        """Boundary TRs of a target's best solution (from the summary, else its result file)."""
        s = self.summary().get(target)
        if s is None:
            return None
        if not s.bounds:
            import gsbs_core as core
            obj = self.load_result(target)
            if obj is None:
                return None
            s.bounds = [int(b) for b in core.boundary_indices(obj.all_bounds[core.best_k(obj)])]
            self.add_summary(target, s)
        return np.asarray(s.bounds, int)

    # ---- stimulus -----------------------------------------------------------

    @property
    def stimulus_dir(self) -> Path:
        return self.path / "stimulus"

    def transcript(self):
        from gsbs_stimulus import Transcript
        p = self.stimulus_dir / "transcript.tsv"
        return Transcript.load(p) if p.exists() else None

    def attach_transcript(self, transcript) -> None:
        self.stimulus_dir.mkdir(exist_ok=True)
        transcript.save(self.stimulus_dir / "transcript.tsv")

    def event_times(self) -> np.ndarray | None:
        p = self.stimulus_dir / "events.txt"
        return np.loadtxt(p, ndmin=1) if p.exists() else None

    def attach_events(self, times_s) -> None:
        self.stimulus_dir.mkdir(exist_ok=True)
        np.savetxt(self.stimulus_dir / "events.txt", np.asarray(times_s, float), fmt="%.3f")

    def add_summary(self, target, s: TargetSummary) -> TargetSummary:
        summary = self.summary()
        summary[target] = s
        path = self.path / "results" / "summary.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(k): v.to_json() for k, v in summary.items()}, indent=1))
        os.replace(tmp, path)
        return s


def summarise(obj, fit_seconds: float, engine: str) -> TargetSummary:
    import gsbs_core as core
    k = core.best_k(obj)
    edges = core.state_edges(obj.all_bounds[k], obj.x.shape[0])
    return TargetSummary(
        nstates=k, n_time=int(obj.x.shape[0]), fit_seconds=float(fit_seconds),
        engine=engine, params=core.fit_params(obj),
        median_duration=float(np.median(np.diff(edges))),
        bounds=[int(b) for b in core.boundary_indices(obj.all_bounds[k])],
        at_limit=core.best_at_limit(obj))
