#!/usr/bin/env python3
"""
Package a fitted Narratives "Sherlock" session as the ready-made demo.

Writes data/demo_sherlock/: the fitted results (gzip-compressed), their
summary, the transcript and the import settings. The voxel data are not
stored: gsbs_nifti.install_demo_session() re-imports the example NIfTI with
the same settings (a few seconds) and checks that the parcel data match the
stored whole-brain fit before adding the results.

Usage:  python tools/make_sherlock_demo.py /path/to/sherlock_gsbs
"""

import gzip
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gsbs_nifti as nifti  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "demo_sherlock"


def main(session_dir: str) -> None:
    s = nifti.Session(session_dir)
    pre = s.meta["preprocessing"]
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "results").mkdir(parents=True)
    for f in sorted((s.path / "results").glob("*.npy")):
        obj = np.load(f, allow_pickle=True).item()
        meta = dict(getattr(obj, "gsbs_gui_meta", {}))
        meta.pop("session", None)                  # the local folder it was fitted in
        obj.gsbs_gui_meta = meta
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=True)
        with gzip.open(OUT / "results" / (f.name + ".gz"), "wb", 9) as dst:
            dst.write(buf.getvalue())
    shutil.copy(s.path / "results" / "summary.json", OUT / "results" / "summary.json")
    for f in ("transcript.tsv", "transcript.json"):
        shutil.copy(s.stimulus_dir / f, OUT / f)
    manifest = {
        "example": "narratives-sherlock",
        "import": {"n_rois": s.spec.n_rois, "yeo_networks": s.spec.yeo_networks, "tr": s.tr,
                   "confounds": None, "detrend": pre["detrend"], "high_pass": pre["high_pass"],
                   "zscore_parcels": pre["zscore_parcels"]},
        "n_time": s.n_time,
        "fitted": sorted(str(k) for k in s.summary()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({size / 1e6:.1f} MB): {len(manifest['fitted'])} fitted targets")


if __name__ == "__main__":
    main(sys.argv[1])
