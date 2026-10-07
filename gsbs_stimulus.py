"""
Stimulus text aligned to the scan, and boundaries derived from it.

A Transcript holds every word with its onset and offset in *scan* time
(seconds from the first volume). Narratives provides Gentle forced alignments
(align.json: word times relative to the start of the audio file, and each
word's character position in the transcript, from which the punctuation after
it is recovered) and an events file giving when the audio file started.

Text is shown against the BOLD signal after shifting every word by an HRF
delay (default 4.5 s): a word heard at t seconds is drawn at the TR containing
t + delay.

Stimulus boundaries (TR at which something new starts, after the same delay):
    sentences  the first word of every sentence after the first
    pauses     the first word after a silence of at least `pause_s` seconds
    events     onsets loaded from a file (seconds from the first volume)
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KINDS = {"sentences": "Sentence starts", "pauses": "Speech pauses", "events": "Loaded events"}


@dataclass
class Transcript:
    words: list                      # display tokens, with trailing punctuation
    onset: np.ndarray                # seconds from the first volume
    offset: np.ndarray
    timed: np.ndarray                # False where the aligner could not place the word
    sentence_end: np.ndarray         # the word ends a sentence
    source: str = ""

    def __len__(self) -> int:
        return len(self.words)

    # ---- I/O ----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["word", "onset", "offset", "timed", "sentence_end"])
            for i in range(len(self)):
                w.writerow([self.words[i], f"{self.onset[i]:.3f}", f"{self.offset[i]:.3f}",
                            int(self.timed[i]), int(self.sentence_end[i])])
        Path(path).with_suffix(".json").write_text(json.dumps({"source": self.source}))

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        """TSV/CSV with columns word, onset[, offset][, timed][, sentence_end] (seconds)."""
        path = Path(path)
        delim = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter=delim))
        if not rows or "word" not in rows[0] or "onset" not in rows[0]:
            raise ValueError("A transcript needs 'word' and 'onset' columns (onsets in seconds "
                             "from the first volume).")
        words = [r["word"] for r in rows]
        onset = np.array([float(r["onset"]) if r["onset"] not in ("", "nan") else np.nan
                          for r in rows])
        offset = np.array([float(r.get("offset") or "nan") for r in rows])
        offset = np.where(np.isnan(offset), onset, offset)
        timed = np.array([bool(int(r["timed"])) if r.get("timed") not in (None, "")
                          else not np.isnan(o) for r, o in zip(rows, onset)])
        if "sentence_end" in rows[0]:
            ends = np.array([bool(int(r["sentence_end"])) for r in rows])
        else:
            ends = np.array([bool(re.search(r"[.!?][\"'”’)]*$", w)) for w in words])
        meta = path.with_suffix(".json")
        source = json.loads(meta.read_text()).get("source", "") if meta.exists() else path.name
        t = cls(words, onset, offset, timed, ends, source)
        t._fill_untimed()
        return t

    @classmethod
    def from_gentle(cls, align_json: str | Path, audio_onset: float, source: str = "") -> "Transcript":
        d = json.loads(Path(align_json).read_text())
        text, ws = d["transcript"], d["words"]
        words, onset, offset, timed, ends = [], [], [], [], []
        for i, w in enumerate(ws):
            nxt = ws[i + 1]["startOffset"] if i + 1 < len(ws) else len(text)
            tail = text[w["endOffset"]:nxt]
            punct = re.match(r"[^\w\s]*", tail.lstrip()).group(0) if tail.strip() else ""
            words.append(text[w["startOffset"]:w["endOffset"]] + punct)
            ok = w.get("case") == "success"
            onset.append(w["start"] + audio_onset if ok else np.nan)
            offset.append(w["end"] + audio_onset if ok else np.nan)
            timed.append(ok)
            ends.append(bool(re.search(r"[.!?]", tail)))
        t = cls(words, np.array(onset), np.array(offset), np.array(timed), np.array(ends), source)
        t._fill_untimed()
        return t

    def _fill_untimed(self) -> None:
        """Place words the aligner missed between their timed neighbours (display only)."""
        idx = np.arange(len(self))
        ok = self.timed & ~np.isnan(self.onset)
        if ok.sum() >= 2 and not ok.all():
            self.onset = np.where(ok, self.onset, np.interp(idx, idx[ok], self.onset[ok]))
            self.offset = np.where(ok, self.offset, np.interp(idx, idx[ok], self.offset[ok]))

    # ---- placing words on the scan -------------------------------------------

    def word_trs(self, tr: float, delay: float) -> np.ndarray:
        """TR at which each word is shown, after the HRF delay."""
        return np.floor((self.onset + delay) / tr).astype(int)

    def boundary_times(self, kind: str, pause_s: float = 1.0) -> np.ndarray:
        """Stimulus boundary times in seconds (before the HRF delay)."""
        if kind == "sentences":
            starts = np.flatnonzero(self.sentence_end[:-1]) + 1
            return self.onset[starts[self.timed[starts]]]
        if kind == "pauses":
            ok = np.flatnonzero(self.timed)
            gaps = self.onset[ok[1:]] - self.offset[ok[:-1]]
            return self.onset[ok[1:][gaps >= pause_s]]
        raise ValueError(kind)


def to_trs(times_s, tr: float, n_time: int, delay: float) -> np.ndarray:
    """Event times (s from the first volume) to boundary TRs after the HRF delay."""
    trs = np.unique(np.floor((np.asarray(times_s, float) + delay) / tr).astype(int))
    return trs[(trs > 0) & (trs < n_time)]


def load_event_times(path: str | Path) -> np.ndarray:
    """Onsets in seconds: a BIDS-style events file (onset column) or one number per line."""
    path = Path(path)
    text = path.read_text().strip().splitlines()
    first = re.split(r"[\t,;]", text[0])
    if "onset" in [c.strip().lower() for c in first]:
        col = [c.strip().lower() for c in first].index("onset")
        return np.array([float(re.split(r"[\t,;]", line)[col]) for line in text[1:] if line.strip()])
    return np.array([float(line.strip().split()[0]) for line in text if line.strip()])


def narratives_audio_onset(events_tsv: str | Path, audio_name: str) -> float:
    """When the story's audio file started, from a Narratives events.tsv (s from the first volume)."""
    with open(events_tsv, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["stim_file"] == audio_name]
    if not rows:
        raise ValueError(f"No events for {audio_name} in {events_tsv}.")
    return min(float(r["onset"]) for r in rows)
