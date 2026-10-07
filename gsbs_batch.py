"""
Batch GSBS over many targets of a session (all ROIs, or a selection).

Each worker is a separate process that reads its ROI's voxels from the
session's memory-mapped voxel file, fits, and saves the result itself; only
small status messages come back. The parent hands out one target at a time, so
pausing lets running fits finish and starts nothing new, cancelling kills the
workers, and results already saved are skipped when a batch is restarted.

Each worker is limited to one BLAS thread, so N workers use N cores rather than
fighting over all of them.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import gsbs_core as core

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
PROGRESS_INTERVAL_S = 1.0


def suggest_workers(n_time: int, max_voxels: int, kmax: int, statewise: bool) -> tuple[int, str]:
    """Number of workers that fits in CPU and memory, and why."""
    cpus = max(1, (os.cpu_count() or 2) - 1)
    per = core.estimate_fit_bytes(n_time, max_voxels, kmax, statewise)
    avail = core.available_bytes()
    if not avail:
        return cpus, f"{cpus} (CPU cores − 1)"
    by_mem = max(1, int(0.7 * avail // per))
    if by_mem < cpus:
        return by_mem, (f"{by_mem} (memory: ≈ {core.format_bytes(per)} each, "
                        f"{core.format_bytes(avail)} free)")
    return cpus, f"{cpus} (CPU cores − 1)"


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

class _WorkerQueue:
    """Receives gsbs_core.run_fit messages inside a worker and forwards a summary."""

    def __init__(self, wid, target, session, out, engine, started):
        self.wid, self.target, self.session, self.out = wid, target, session, out
        self.engine, self.started = engine, started
        self._last = 0.0

    def put(self, msg):
        kind = msg[0]
        if kind == "progress":
            now = time.monotonic()
            if now - self._last >= PROGRESS_INTERVAL_S:
                self._last = now
                self.out.put(("progress", self.wid, self.target, msg[1], msg[2]))
        elif kind == "done":
            obj = msg[1]
            seconds = time.monotonic() - self.started
            obj.gsbs_gui_meta = self.session.target_meta(self.target) | {"fit_seconds": seconds}
            path = self.session.result_path(self.target)
            tmp = path.with_name(path.stem + f".w{self.wid}.tmp.npy")
            core.save_object(obj, tmp)
            os.replace(tmp, path)
            from gsbs_nifti import summarise
            s = summarise(obj, seconds, self.engine)
            self.out.put(("finished", self.wid, self.target, s.to_json()))
        else:
            self.out.put(("failed", self.wid, self.target, msg[1]))


def _worker_main(wid, session_dir, inbox, out, engine, params):
    from gsbs_nifti import Session
    try:
        session = Session(session_dir)
    except BaseException:                                   # noqa: BLE001
        out.put(("failed", wid, None, traceback.format_exc()))
        return
    out.put(("ready", wid))
    while True:
        target = inbox.get()
        if target is None:
            return
        started = time.monotonic()
        try:
            x = session.data(target)
            if x.shape[1] < 2:
                raise ValueError(f"only {x.shape[1]} voxel(s); GSBS needs at least 2")
            p = dict(params)
            p["kmax"] = min(p["kmax"], x.shape[0] - 1)
            out.put(("started", wid, target, x.shape[1]))
            core.run_fit(_WorkerQueue(wid, target, session, out, engine, started), engine, x, p)
        except BaseException:                                # noqa: BLE001
            out.put(("failed", wid, target, traceback.format_exc()))


# ---------------------------------------------------------------------------
# Parent-side manager
# ---------------------------------------------------------------------------

@dataclass
class _Running:
    target: object
    started: float
    done: int = 0
    total: int = 0


@dataclass
class BatchStatus:
    total: int
    finished: int
    failed: int
    running: dict                    # worker id -> (target, fraction done, elapsed s)
    pending: int
    paused: bool
    eta_seconds: float | None
    mean_seconds: float | None
    events: list = field(default_factory=list)   # ("finished", target, summary) / ("failed", target, msg)


class BatchRunner:
    def __init__(self, session_dir: str, targets: list, engine: str, params: dict,
                 n_workers: int, prior_seconds: float | None = None):
        self.session_dir = str(session_dir)
        self.targets = list(targets)
        self.engine, self.params = engine, dict(params)
        self.n_workers = max(1, min(int(n_workers), len(self.targets) or 1))
        self.prior_seconds = prior_seconds       # e.g. the whole-brain fit time, before any ROI finishes
        self._pending = deque(self.targets)
        self._running: dict[int, _Running] = {}
        self._durations: list[float] = []
        self._failed: dict = {}
        self._idle: set[int] = set()
        self._procs: dict[int, mp.Process] = {}
        self._inboxes: dict = {}
        self._out = None
        self.paused = False
        self.started_at = None

    # ---- control -------------------------------------------------------------

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._out = ctx.Queue()
        saved = {v: os.environ.get(v) for v in _THREAD_VARS}
        try:
            for v in _THREAD_VARS:
                os.environ[v] = "1"
            for wid in range(self.n_workers):
                inbox = ctx.Queue()
                proc = ctx.Process(target=_worker_main, daemon=True,
                                   args=(wid, self.session_dir, inbox, self._out,
                                         self.engine, self.params))
                proc.start()
                self._procs[wid], self._inboxes[wid] = proc, inbox
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
        self.started_at = time.monotonic()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        for wid in list(self._idle):
            self._dispatch(wid)

    def cancel(self) -> None:
        for proc in self._procs.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs.values():
            proc.join(2)
        for q in list(self._inboxes.values()) + ([self._out] if self._out else []):
            q.cancel_join_thread()
            q.close()
        self._procs.clear()
        self._inboxes.clear()
        self._running.clear()
        self._idle.clear()

    @property
    def active(self) -> bool:
        return bool(self._procs)

    @property
    def finished_all(self) -> bool:
        return not self._pending and not self._running

    # ---- polling -------------------------------------------------------------

    def _dispatch(self, wid: int) -> None:
        if self.paused or not self._pending:
            self._idle.add(wid)
            if not self._pending and not self._running:
                self._shutdown()
            return
        self._idle.discard(wid)
        target = self._pending.popleft()
        self._running[wid] = _Running(target, time.monotonic())
        self._inboxes[wid].put(target)

    def _shutdown(self) -> None:
        for wid, inbox in self._inboxes.items():
            inbox.put(None)
        for proc in self._procs.values():
            proc.join(5)
        self._procs.clear()
        self._inboxes.clear()

    def poll(self) -> BatchStatus:
        events = []
        while self._out is not None and self._procs:
            try:
                msg = self._out.get_nowait()
            except queue_mod.Empty:
                break
            kind, wid = msg[0], msg[1]
            if kind == "ready":
                self._dispatch(wid)
            elif kind == "started":
                pass
            elif kind == "progress":
                if wid in self._running:
                    self._running[wid].done, self._running[wid].total = msg[3], msg[4]
            elif kind in ("finished", "failed"):
                target = msg[2]
                run = self._running.pop(wid, None)
                if kind == "finished":
                    self._durations.append(msg[3]["fit_seconds"])
                    events.append(("finished", target, msg[3]))
                else:
                    self._failed[target] = msg[3]
                    events.append(("failed", target, msg[3]))
                if run is not None or target is None:
                    if target is None:              # the worker could not even start
                        self._procs.pop(wid, None)
                    else:
                        self._dispatch(wid)
        self._check_dead_workers(events)
        return self._status(events)

    def _check_dead_workers(self, events) -> None:
        for wid, proc in list(self._procs.items()):
            if not proc.is_alive():
                run = self._running.pop(wid, None)
                if run is not None:
                    hint = " (possibly out of memory)" if proc.exitcode in (-9, 137) else ""
                    msg = f"worker exited with code {proc.exitcode}{hint}"
                    self._failed[run.target] = msg
                    events.append(("failed", run.target, msg))
                self._procs.pop(wid)
                self._inboxes.pop(wid, None)
                self._idle.discard(wid)

    def _status(self, events) -> BatchStatus:
        now = time.monotonic()
        mean = float(np.mean(self._durations)) if self._durations else self.prior_seconds
        if mean is None:                         # nothing finished yet: project running fits
            projected = [(now - r.started) / (r.done / r.total)
                         for r in self._running.values() if r.total and r.done / r.total > 0.05]
            mean = float(np.median(projected)) if projected else None
        running = {}
        remaining_running = []
        for wid, r in self._running.items():
            elapsed = now - r.started
            frac = r.done / r.total if r.total else 0.0
            running[wid] = (r.target, frac, elapsed)
            if frac > 0.05:
                remaining_running.append(elapsed / frac - elapsed)
            elif mean:
                remaining_running.append(max(0.0, mean - elapsed))
        eta = None
        if mean and (self._procs or self._pending):
            workers = max(1, len(self._procs))
            eta = (sum(remaining_running) + len(self._pending) * mean) / workers
            if remaining_running:
                eta = max(eta, max(remaining_running))
        return BatchStatus(
            total=len(self.targets), finished=len(self._durations), failed=len(self._failed),
            running=running, pending=len(self._pending), paused=self.paused,
            eta_seconds=eta, mean_seconds=mean, events=events)
