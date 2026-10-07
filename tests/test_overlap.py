"""Boundary overlap measures and stimulus transcripts."""

import json

import numpy as np
import pytest

import gsbs_overlap as ov
import gsbs_stimulus as stim


class TestOverlap:

    def test_matches_neuro_tools_formula(self):
        ref, test, T = np.array([10, 20, 30, 40]), np.array([10, 20, 33, 50, 60]), 100
        r = ov.overlap(ref, test, T, n_perm=200)
        O, E = 2, 4 * 5 / 100
        assert r.observed == O and np.isclose(r.expected, E)
        assert np.isclose(r.oa, (O - E) / (5 - E))
        assert np.isclose(r.or_, (O - E) / (4 - E))

    def test_identical_sets(self):
        b = np.array([7, 19, 33, 52, 70, 88])
        r = ov.overlap(b, b, 100, n_perm=500)
        assert r.oa == 1 and r.or_ == 1 and r.p < 0.01

    def test_tolerance_matches_once(self):
        assert ov.count_matches([10, 11], [10], 1) == 1
        assert ov.count_matches([10], [9, 11], 1) == 1          # one reference boundary, used once
        assert ov.count_matches([10, 20], [11, 21], 1) == 2
        assert ov.count_matches([10, 20], [12], 1) == 0

    def test_shuffle_keeps_durations(self):
        rng = np.random.default_rng(0)
        b = np.array([5, 15, 40, 41, 90])
        s = ov.shuffle_states(b, 100, rng)
        assert sorted(np.diff(np.r_[0, s, 100])) == sorted(np.diff(np.r_[0, b, 100]))

    def test_p_counts_ties(self):
        # every shuffle of a single-state-change sequence gives the same overlap count
        r = ov.overlap([50], [50], 100, n_perm=99)
        assert 0 < r.p <= 1


class TestTranscript:

    @pytest.fixture
    def gentle(self, tmp_path):
        text = "Hello there. How are you? Fine."
        words, pos = [], 0
        times = [(0.0, 0.4), (0.5, 0.9), (2.5, 2.7), None, (3.0, 3.2), (6.0, 6.4)]
        for i, tok in enumerate(["Hello", "there", "How", "are", "you", "Fine"]):
            a = text.index(tok, pos)
            pos = a + len(tok)
            w = {"word": tok, "alignedWord": tok.lower(), "startOffset": a, "endOffset": pos,
                 "case": "success" if times[i] else "not-found-in-audio", "phones": []}
            if times[i]:
                w["start"], w["end"] = times[i]
            words.append(w)
        p = tmp_path / "align.json"
        p.write_text(json.dumps({"transcript": text, "words": words}))
        return p

    def test_gentle_punctuation_timing_and_untimed(self, gentle, tmp_path):
        t = stim.Transcript.from_gentle(gentle, audio_onset=10.0)
        assert t.words == ["Hello", "there.", "How", "are", "you?", "Fine."]
        assert t.sentence_end.tolist() == [False, True, False, False, True, True]
        assert t.onset[0] == 10.0 and not t.timed[3]
        assert 12.5 < t.onset[3] < 13.0                             # placed between neighbours
        np.testing.assert_allclose(t.boundary_times("sentences"), [12.5, 16.0])
        np.testing.assert_allclose(t.boundary_times("pauses", 1.0), [12.5, 16.0])
        t.save(tmp_path / "t.tsv")
        back = stim.Transcript.load(tmp_path / "t.tsv")
        assert back.words == t.words and np.allclose(back.onset, t.onset)

    def test_to_trs_applies_delay(self):
        trs = stim.to_trs([0.0, 3.0, 3.1, 1000.0], tr=1.5, n_time=100, delay=4.5)
        assert trs.tolist() == [3, 5]                                 # 0 -> TR 3; 3.0 and 3.1 -> TR 5

    def test_event_files(self, tmp_path):
        (tmp_path / "e.tsv").write_text("onset\tduration\ttrial_type\n4.5\t2\ta\n30\t2\tb\n")
        (tmp_path / "e.txt").write_text("12\n40.5\n")
        assert stim.load_event_times(tmp_path / "e.tsv").tolist() == [4.5, 30.0]
        assert stim.load_event_times(tmp_path / "e.txt").tolist() == [12.0, 40.5]

    def test_narratives_audio_onset(self, tmp_path):
        p = tmp_path / "ev.tsv"
        p.write_text("onset\tduration\ttrial_type\tstim_file\n4.5\t25.0\tmusic\tsherlock_audio.wav\n"
                     "33.5\t1052.0\tstory\tsherlock_audio.wav\n")
        assert stim.narratives_audio_onset(p, "sherlock_audio.wav") == 4.5


class TestGrid:

    def test_single_reference_matches_overlap(self):
        rng = np.random.default_rng(0)
        ref = np.sort(rng.choice(np.arange(1, 400), 30, replace=False))
        test = np.sort(rng.choice(np.arange(1, 400), 40, replace=False))
        a = ov.overlap(ref, test, 400, n_perm=300, seed=3)
        g = ov.grid_overlap([ref], test, 400, n_perm=300, seed=3)
        assert a.observed == g.observed[0] and np.isclose(a.or_, g.or_[0]) and np.isclose(a.p, g.p[0])

    def test_best_delay_and_corrected_p(self):
        rng = np.random.default_rng(1)
        T = 600
        stim = np.sort(rng.choice(np.arange(5, T - 10), 50, replace=False))
        refs = [stim + d for d in range(6)]
        g = ov.grid_overlap(refs, stim + 3, T, n_perm=300)
        assert g.best == 3 and g.p_best < 0.01
        # unrelated segmentations: the search-corrected p keeps false positives near 5%,
        # the uncorrected p at the best delay does not
        fp_corr = fp_naive = 0
        for s in range(60):
            test = np.sort(rng.choice(np.arange(1, T), 60, replace=False))
            r = ov.grid_overlap(refs, test, T, n_perm=150, seed=s)
            fp_corr += r.p_best < 0.05
            fp_naive += r.p[r.best] < 0.05
        assert fp_corr <= 8 and fp_naive > fp_corr

    def test_fdr(self):
        q = ov.fdr_bh([0.01, 0.04, 0.03, np.nan, 0.5])
        assert np.isnan(q[3]) and np.isclose(q[0], 0.04) and q[4] == 0.5
