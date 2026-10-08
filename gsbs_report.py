"""
Methods and Results text, and figure captions, generated from an analysis
configuration and the results the replication scripts produce.

Text follows the conventions of an empirical paper: Methods in the past tense
and passive voice, Results in the past tense without interpretation, every
figure cited in order, statistics with test, value, p, correction and n. Items
the tool cannot know (participants, acquisition, code repository) are marked
[TODO: ...] rather than guessed. British spelling throughout; no em dashes.

Paragraphs are built as lists of sentences with markers, and rendered to
Markdown and to LaTeX (one sentence per line, \\citep keys, \\todo{} markers):
    {cite:Key1,Key2}      parenthetical citation, {citet:Key} narrative citation
    {fig:id}              figure reference, {fig:id:A} with a panel
"""

from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

REFS = {
    "Geerligs2021": ("Geerligs et al., 2021", """@article{Geerligs2021,
  author = {Geerligs, Linda and van Gerven, Marcel and G{\\"u}{\\c{c}}l{\\"u}, Umut},
  title = {Detecting neural state transitions underlying event segmentation},
  journal = {NeuroImage}, year = {2021}, volume = {236}, pages = {118085},
  doi = {10.1016/j.neuroimage.2021.118085}}"""),
    "Geerligs2022": ("Geerligs et al., 2022", """@article{Geerligs2022,
  author = {Geerligs, Linda and G{\\"o}z{\\"u}kara, Dora and Oetringer, Djamari and Campbell, Karen and van Gerven, Marcel and G{\\"u}{\\c{c}}l{\\"u}, Umut},
  title = {A partially nested cortical hierarchy of neural states underlies event segmentation in the human brain},
  journal = {bioRxiv}, year = {2022}, doi = {10.1101/2021.02.05.429165},
  note = {TODO: check whether to cite the published version}}"""),
    "Schaefer2018": ("Schaefer et al., 2018", """@article{Schaefer2018,
  author = {Schaefer, Alexander and Kong, Ru and Gordon, Evan M. and Laumann, Timothy O. and Zuo, Xi-Nian and Holmes, Avram J. and Eickhoff, Simon B. and Yeo, B. T. Thomas},
  title = {Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity {MRI}},
  journal = {Cerebral Cortex}, year = {2018}, volume = {28}, number = {9}, pages = {3095--3114},
  doi = {10.1093/cercor/bhx179}}"""),
    "Yeo2011": ("Yeo et al., 2011", """@article{Yeo2011,
  author = {Yeo, B. T. Thomas and Krienen, Fenna M. and Sepulcre, Jorge and Sabuncu, Mert R. and Lashkari, Danial and Hollinshead, Marisa and Roffman, Joshua L. and Smoller, Jordan W. and Z{\\"o}llei, Lilla and Polimeni, Jonathan R. and Fischl, Bruce and Liu, Hesheng and Buckner, Randy L.},
  title = {The organization of the human cerebral cortex estimated by intrinsic functional connectivity},
  journal = {Journal of Neurophysiology}, year = {2011}, volume = {106}, number = {3}, pages = {1125--1165},
  doi = {10.1152/jn.00338.2011}}"""),
    "Abraham2014": ("Abraham et al., 2014", """@article{Abraham2014,
  author = {Abraham, Alexandre and Pedregosa, Fabian and Eickenberg, Michael and Gervais, Philippe and Mueller, Andreas and Kossaifi, Jean and Gramfort, Alexandre and Thirion, Bertrand and Varoquaux, Ga{\\"e}l},
  title = {Machine learning for neuroimaging with scikit-learn},
  journal = {Frontiers in Neuroinformatics}, year = {2014}, volume = {8}, pages = {14},
  doi = {10.3389/fninf.2014.00014}}"""),
    "Pedregosa2011": ("Pedregosa et al., 2011", """@article{Pedregosa2011,
  author = {Pedregosa, Fabian and others},
  title = {Scikit-learn: Machine learning in {P}ython},
  journal = {Journal of Machine Learning Research}, year = {2011}, volume = {12}, pages = {2825--2830}}"""),
    "Nastase2021": ("Nastase et al., 2021", """@article{Nastase2021,
  author = {Nastase, Samuel A. and Liu, Yun-Fei and Hillman, Hanna and Zadbood, Asieh and Hasenfratz, Liat and Keshavarzian, Neggin and Chen, Janice and Honey, Christopher J. and Yeshurun, Yaara and Regev, Mor and others},
  title = {The {``Narratives''} {fMRI} dataset for evaluating models of naturalistic language comprehension},
  journal = {Scientific Data}, year = {2021}, volume = {8}, pages = {250},
  doi = {10.1038/s41597-021-01033-3}}"""),
    "Richardson2018": ("Richardson et al., 2018", """@article{Richardson2018,
  author = {Richardson, Hilary and Lisandrelli, Grace and Riobueno-Naylor, Alexa and Saxe, Rebecca},
  title = {Development of the social brain from age three to twelve years},
  journal = {Nature Communications}, year = {2018}, volume = {9}, pages = {1027},
  doi = {10.1038/s41467-018-03399-2}}"""),
    "Esteban2019": ("Esteban et al., 2019", """@article{Esteban2019,
  author = {Esteban, Oscar and Markiewicz, Christopher J. and Blair, Ross W. and others},
  title = {{fMRIPrep}: a robust preprocessing pipeline for functional {MRI}},
  journal = {Nature Methods}, year = {2019}, volume = {16}, pages = {111--116},
  doi = {10.1038/s41592-018-0235-4}}"""),
    "Fischl2012": ("Fischl, 2012", """@article{Fischl2012,
  author = {Fischl, Bruce}, title = {{FreeSurfer}},
  journal = {NeuroImage}, year = {2012}, volume = {62}, number = {2}, pages = {774--781},
  doi = {10.1016/j.neuroimage.2012.01.021}}"""),
    "Benjamini1995": ("Benjamini and Hochberg, 1995", """@article{Benjamini1995,
  author = {Benjamini, Yoav and Hochberg, Yosef},
  title = {Controlling the false discovery rate: a practical and powerful approach to multiple testing},
  journal = {Journal of the Royal Statistical Society: Series B}, year = {1995}, volume = {57}, number = {1}, pages = {289--300}}"""),
    "Virtanen2020": ("Virtanen et al., 2020", """@article{Virtanen2020,
  author = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and others},
  title = {{SciPy} 1.0: fundamental algorithms for scientific computing in {P}ython},
  journal = {Nature Methods}, year = {2020}, volume = {17}, pages = {261--272},
  doi = {10.1038/s41592-019-0686-2}}"""),
    "Rousseeuw1987": ("Rousseeuw, 1987", """@article{Rousseeuw1987,
  author = {Rousseeuw, Peter J.},
  title = {Silhouettes: a graphical aid to the interpretation and validation of cluster analysis},
  journal = {Journal of Computational and Applied Mathematics}, year = {1987}, volume = {20}, pages = {53--65},
  doi = {10.1016/0377-0427(87)90125-7}}"""),
    "Ester1996": ("Ester et al., 1996", """@inproceedings{Ester1996,
  author = {Ester, Martin and Kriegel, Hans-Peter and Sander, J{\\"o}rg and Xu, Xiaowei},
  title = {A density-based algorithm for discovering clusters in large spatial databases with noise},
  booktitle = {Proceedings of the Second International Conference on Knowledge Discovery and Data Mining}, year = {1996}, pages = {226--231}}"""),
    "Gentle": ("Ochshorn and Hawkins, 2016", """@misc{Gentle,
  author = {Ochshorn, Robert M. and Hawkins, Max}, title = {Gentle: a robust yet lenient forced aligner built on {Kaldi}},
  year = {2016}, howpublished = {\\url{https://lowerquality.com/gentle}}}"""),
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_p(p: float, n_perm: int | None = None) -> str:
    if p is None or np.isnan(p):
        return "p = n/a"
    floor = 1 / (n_perm + 1) if n_perm else 0
    if n_perm and p <= floor + 1e-12:
        return f"p < {1 / n_perm:.3g}" if n_perm >= 1000 else f"p = {p:.3g}"
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def fmt_s(seconds: float | None, trs: float) -> str:
    if seconds is None:
        return f"{trs:g} TRs"
    return f"{seconds:.3g} s ({trs:g} TRs)"


_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def plural(n: int, word: str, words: bool = False) -> str:
    """'3 states', '1 state', 'boundaries'; words=True spells out 0-9 (for sentence starts)."""
    if n == 1:
        form = word
    elif word.endswith("y") and word[-2] not in "aeiou":
        form = word[:-1] + "ies"
    else:
        form = word + "s"
    num = _WORDS[n] if words and 0 <= n < 10 else str(n)
    return f"{num} {form}"


def and_list(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_order(cfg: dict, res: dict) -> list[str]:
    order = ["segmentation"]
    if res.get("boundary_example"):
        order.append("boundary")
    if res.get("timescales") and len(res["timescales"]["rois"]) >= 2:
        order.append("timescales")
    if res.get("stimulus_overlap"):
        order.append("stimulus_overlap")
    if res.get("overlap_map"):
        order.append("overlap_map")
    if res.get("clusters"):
        order.append("clusters")
    return order


class Doc:
    """Paragraph builder with citation and figure markers."""

    def __init__(self, figures: list[str]):
        self.figures = figures
        self.paragraphs: list[tuple[str | None, list[str]]] = []
        self.cited: list[str] = []

    def section(self, title: str, sentences: list[str]) -> None:
        sentences = [s for s in sentences if s]
        if sentences:
            self.paragraphs.append((title, sentences))

    def _citet(self, key: str, tex: bool) -> str:
        key = key.strip()
        if key not in self.cited:
            self.cited.append(key)
        if tex:
            return f"\\citet{{{key}}}"
        name, _, year = REFS[key][0].rpartition(", ")
        return f"{name} ({year})"

    def _cite(self, keys: str, tex: bool) -> str:
        ks = [k.strip() for k in keys.split(",")]
        for k in ks:
            if k not in self.cited:
                self.cited.append(k)
        if tex:
            return f"\\citep{{{','.join(ks)}}}"
        return "(" + "; ".join(REFS[k][0] for k in ks) + ")"

    def _fig(self, ref: str, tex: bool) -> str:
        fid, _, panel = ref.partition(":")
        n = self.figures.index(fid) + 1
        if tex:
            return f"Fig.~\\ref{{fig:{fid}}}{panel}"
        return f"Fig. {n}{panel}"

    def render(self, sentence: str, tex: bool) -> str:
        s = re.sub(r"\{citet:([^}]+)\}", lambda m: self._citet(m.group(1), tex), sentence)
        s = re.sub(r"\{cite:([^}]+)\}", lambda m: self._cite(m.group(1), tex), s)
        s = re.sub(r"\{fig:([^}]+)\}", lambda m: self._fig(m.group(1), tex), s)
        if tex:
            s = re.sub(r"\[TODO: ([^\]]+)\]", lambda m: f"\\todo{{{m.group(1)}}}", s)
            s = s.replace("%", "\\%").replace("\u2212", "$-$").replace("\u00b1", "$\\pm$")
            s = s.replace("\u2264", "$\\leq$").replace("\u00d7", "$\\times$")
        return s

    def markdown(self, heading: str) -> str:
        out = [f"## {heading}", ""]
        for title, sentences in self.paragraphs:
            if title:
                out += [f"### {title}", ""]
            out += [" ".join(self.render(s, False) for s in sentences), ""]
        return "\n".join(out)

    def latex(self, heading: str) -> str:
        out = [f"\\section*{{{heading}}}", ""]
        for title, sentences in self.paragraphs:
            if title:
                out += [f"\\subsection*{{{title}}}", ""]
            out += [self.render(s, True) for s in sentences] + [""]
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

DATA_TEXT = {
    "narratives-sherlock": [
        "We analysed one participant (sub-151) from the publicly available Narratives dataset {cite:Nastase2021}, who listened to an 18-minute spoken recall of the first episode of the BBC series Sherlock.",
        "Functional images were acquired with a repetition time (TR) of 1.5 s; the run comprised {n_time} volumes.",
        "We used the distributed preprocessed images (afni-nosmooth derivatives), which had been preprocessed with fMRIPrep {cite:Esteban2019}, spatially normalised to MNI152NLin2009cAsym space, and cleaned of six head motion parameters, the first five principal components of cerebrospinal fluid and white matter signals, a cosine basis high-pass filter (cutoff 128 s) and first- and second-order polynomial trends {cite:Nastase2021}.",
    ],
    "narratives-tunnel": [
        "We analysed one participant (sub-041) from the publicly available Narratives dataset {cite:Nastase2021}, who listened to the science-fiction story Tunnel Under the World by Frederik Pohl, as broadcast in the X Minus One radio series.",
        "Functional images were acquired with a TR of 1.5 s; the run comprised {n_time} volumes.",
        "We used the distributed preprocessed images (afni-nosmooth derivatives), which had been preprocessed with fMRIPrep {cite:Esteban2019}, normalised to MNI152NLin2009cAsym space and cleaned of motion, cerebrospinal fluid and white matter components, low-frequency drifts and polynomial trends {cite:Nastase2021}.",
    ],
    "narratives-21styear": [
        "We analysed one participant (sub-257) from the publicly available Narratives dataset {cite:Nastase2021}, who listened to the 56-minute story The 21st Year by Christina Lazaridi.",
        "Functional images were acquired with a TR of 1.5 s; the run comprised {n_time} volumes.",
        "We used the distributed preprocessed images (afni-nosmooth derivatives), which had been preprocessed with fMRIPrep {cite:Esteban2019}, normalised to MNI152NLin2009cAsym space and cleaned of motion, cerebrospinal fluid and white matter components, low-frequency drifts and polynomial trends {cite:Nastase2021}.",
    ],
    "pixar": [
        "We analysed one adult participant from the developmental fMRI dataset of {citet:Richardson2018}, as distributed with nilearn {cite:Abraham2014}, who watched the short animated film Partly Cloudy (Pixar Animation Studios).",
        "Functional images were acquired with a TR of 2 s; the run comprised {n_time} volumes and had been preprocessed with fMRIPrep {cite:Esteban2019} and normalised to MNI152NLin2009cAsym space.",
    ],
}


def _test_sentence(res: dict, cfg: dict) -> str:
    refs = []
    if res.get("stimulus_overlap"):
        refs.append("the stimulus boundaries")
    om = cfg.get("overlap_map")
    if om and om["reference"]["kind"] == "stimulus" and not refs:
        refs.append("the stimulus boundaries")
    if om and om["reference"]["kind"] == "neural":
        refs.append("the whole-brain boundaries (for the parcel analyses)")
    return (f"Neural state boundaries were always the test segmentation, and {and_list(refs)} "
            f"{'the reference' if len(refs) == 1 else 'the references'}.")


def _cluster_choice(cl: dict) -> str:
    """How the number of clusters (or the DBSCAN parameters) was chosen."""
    scan = cl.get("scan") or {}
    if cl.get("selected_by_silhouette") and cl["method"] == "dbscan" and scan:
        return (f"The neighbourhood radius and minimum number of states were chosen as the combination maximising the mean silhouette {{cite:Rousseeuw1987}}, computed with the same distance as the clustering and counting states left unclustered as 0, over {scan['n_eps']} radii from {scan['eps'][0]:.3g} to {scan['eps'][1]:.3g} and minimum numbers of {and_list([str(v) for v in scan['min_samples']])} states.")
    if cl.get("selected_by_silhouette") and scan:
        lo, hi = scan["n_clusters"]
        return (f"The number of clusters was chosen as the one maximising the mean silhouette {{cite:Rousseeuw1987}} across solutions with {lo} to {hi} clusters, computed with the same distance as the clustering.")
    if cl["method"] == "dbscan":
        return "[TODO: justify the choice of the DBSCAN neighbourhood radius and minimum number of states.]"
    return "[TODO: justify the number of clusters.]"


def methods(cfg: dict, res: dict) -> Doc:
    figs = figure_order(cfg, res)
    d = Doc(figs)
    data = cfg["data"]
    seg = res["segmentation"]
    tr = data.get("tr")
    imp = data.get("import") or {}
    key = data.get("example")

    if key in DATA_TEXT:
        d.section("Data", [s.replace("{n_time}", str(seg["n_time"])) for s in DATA_TEXT[key]])
    elif data["kind"] == "nifti":
        d.section("Data", [
            f"We analysed a functional MRI time series of {seg['n_time']} volumes"
            + (f" acquired with a TR of {tr:g} s" if tr else "") + " in MNI152 space.",
            "[TODO: participants, stimulus, acquisition parameters and preprocessing of the input data.]",
        ])
    else:
        d.section("Data", [
            f"We analysed a time series of {seg['n_time']} timepoints and {seg['n_columns']} "
            f"{data.get('columns', 'columns')}" + (f" (TR = {tr:g} s)." if tr else "."),
            "[TODO: origin and preprocessing of the input data.]",
        ])

    if data["kind"] == "nifti":
        spec = data["atlas"]
        clean = []
        cols = data.get("confound_columns") or []
        if imp.get("confounds"):
            clean.append(f"regressed against {len(cols)} confound time series from the preprocessing output ({', '.join(cols)})"
                         if cols else "regressed against confound time series [TODO: list the confound regressors]")
        if imp.get("high_pass"):
            clean.append(f"high-pass filtered at {imp['high_pass']:g} Hz")
        if imp.get("detrend"):
            clean.append("linearly detrended")
        clean_s = (", ".join(clean[:-1]) + (" and " if len(clean) > 1 else "") + clean[-1]) if clean else ""
        d.section("Parcellation and time series extraction", [
            f"Voxels were assigned to the {spec['n_rois']} parcels of the "
            + (f"{data['atlas_resolution_mm']} mm, " if data.get("atlas_resolution_mm") else "")
            + f"{spec['yeo_networks']}-network version of the Schaefer atlas {{cite:Schaefer2018}}, resampled to the functional images with nearest-neighbour interpolation, using nilearn {{cite:Abraham2014}}.",
            (f"Each voxel time series was {clean_s} and then z-scored over time." if clean_s
             else "Each voxel time series was z-scored over time."),
            "Parcel time series were obtained by averaging the z-scored voxel time series within each parcel"
            + (", and were z-scored again." if imp.get("zscore_parcels") else "."),
            "Voxels with a constant signal were excluded"
            + (f", and {plural(data['missing_parcels'], 'parcel')} that contained no voxels after resampling {'was' if data['missing_parcels'] == 1 else 'were'} left out." if data.get("missing_parcels") else "."),
        ])

    p = seg["params"]
    whole = cfg["segmentation"]["target"]
    unit = {"whole_brain": "parcel", "data": data.get("columns", "column").rstrip("s")}.get(whole, "voxel")
    ft = p["finetune"]
    order_s = ("from the weakest to the strongest" if p.get("finetune_order", True)
               else "from the strongest to the weakest")
    ft_s = ("no finetuning of boundary locations" if ft == 0 else
            "finetuning of every boundary over all timepoints" if ft < 0 else
            f"finetuning of each boundary location within {plural(ft, 'TR')} of its initial position")
    d.section("Neural state segmentation", [
        f"Neural state boundaries were identified in the {unit} time series with greedy state boundary search (GSBS), following {{citet:Geerligs2021}}.",
        "GSBS places boundaries one at a time where they best separate stable activity patterns, and determines the optimal number of states with the t-distance, the t-statistic comparing the correlations between timepoints within the same state with the correlations between timepoints in consecutive states {cite:Geerligs2021}.",
        (f"We used the improved version of the algorithm {{cite:Geerligs2022}}, with statewise detection, which allows two boundaries delimiting a new state to be added in a single step, and {ft_s}" + (f", with boundaries finetuned in order {order_s}." if ft else ".")
         if p["statewise_detection"] else
         f"We used the original formulation, adding one boundary per step, with {ft_s}"
         + (f", with boundaries finetuned in order {order_s}." if ft else ".")),
        (f"Candidate boundaries were pre-selected within blocks of {p['blocksize']} timepoints (the blocksize parameter, default 50)."
         if p.get("blocksize", 50) != 50 else ""),
        f"Solutions with up to {p['kmax']} states were evaluated"
        + (" (half the number of timepoints)" if abs(p["kmax"] - seg["n_time"] / 2) <= 1 else "")
        + (f", and only correlations between timepoints at least {p['dmin']} TRs apart entered the t-distance." if p.get("dmin", 1) > 1 else "."),
        (f"We used fast_GSBS, an accelerated implementation in the statesegmentation package (version {cfg['package']['software'].get('statesegmentation', '[TODO: version]')}) that returns results identical to the reference implementation."
         if cfg["fits"].get("engine", seg["engine"]) == "fast_GSBS" else
         f"We used the reference implementation in the statesegmentation package (version {cfg['package']['software'].get('statesegmentation', '[TODO: version]')})."),
    ])

    if res.get("timescales"):
        ts = res["timescales"]
        n_fit = len(ts["rois"]) + len(ts["excluded"])
        import json as _json
        roi_params = {_json.dumps(v, sort_keys=True) for k_, v in cfg["fits"].get("params", {}).items() if k_ != "whole_brain"}
        same = len(roi_params) <= 1 and (not roi_params or _json.dumps(p, sort_keys=True) in roi_params)
        d.section("Region-of-interest segmentation", [
            f"The same procedure, {'with identical parameters' if same else 'with the parameters listed in analysis_config.json [TODO: report the parameters that differ between parcels]'}, was applied separately to the voxel time series of {plural(n_fit, 'parcel')}"
            + (" [TODO: state how these parcels were selected]." if n_fit < cfg["data"]["atlas"]["n_rois"] else "."),
            "Parcels in which the t-distance was still increasing at kmax, so that no optimum was found within the search range, were excluded from further analysis"
            + (f" ({plural(len(ts['excluded']), 'parcel')})." if ts["excluded"] else " (none met this criterion)."),
            "For each parcel, the timescale of its states was summarised by the median duration of the states in its optimal solution.",
        ])

    st = cfg.get("stimulus")
    if st:
        kind = st["boundaries"]
        if kind == "events":
            what = "the onsets of the annotated events [TODO: describe the event annotation]"
        elif kind == "sentences":
            what = "the onset of the first word of every sentence after the first"
        else:
            what = f"the onset of the first word after every silence of at least {st['pause_s']:g} s"
        sents = []
        if cfg.get("transcript"):
            sents.append("Word onsets were taken from the forced alignment of the story transcript to the audio distributed with the dataset {cite:Gentle,Nastase2021}, and placed on the scan time course using the onset of the audio recorded in the participant's events file."
                         if str(key).startswith("narratives") else
                         "Word onsets were taken from a time-aligned transcript of the stimulus [TODO: describe how the transcript was aligned].")
        si = res.get("stimulus_info") or {}
        sents.append(f"Stimulus boundaries were defined as {what}.")
        if si.get("n_untimed"):
            sents.append(f"Words that the aligner could not place in the audio ({si['n_untimed']} of {si['n_words']}, {100 * si['n_untimed'] / si['n_words']:.1f}%) could not start a boundary.")
        sents.append(f"To account for the haemodynamic delay, every stimulus boundary was shifted by {st['delay_s']:g} s and assigned to the TR containing the shifted time.")
        if si:
            notes = []
            if si.get("n_merged"):
                notes.append(f"{plural(si['n_merged'], 'boundary', words=True)} that fell in the same TR as another {'was' if si['n_merged'] == 1 else 'were'} counted once")
            if si.get("n_outside"):
                notes.append(f"{plural(si['n_outside'], 'boundary', words=not notes)} that fell outside the scan after the shift {'was' if si['n_outside'] == 1 else 'were'} discarded")
            sents.append((cap(and_list(notes)) + ", " if notes else "") + f"{'leaving' if notes else 'This gave'} {plural(si['n_trs'], 'stimulus boundary')} in total.")
        d.section("Stimulus boundaries", sents)

    if res.get("stimulus_overlap") or res.get("overlap_map"):
        so = res.get("stimulus_overlap") or {}
        tol = (st or {}).get("tolerance", 0)
        sents = [
            "Overlap between two sets of boundaries was quantified following {citet:Geerligs2022}.",
            "The observed overlap O was the number of boundaries shared by both sets"
            + (f" (within {plural(tol, 'TR')})" if tol else " (at the same TR)")
            + ", and the overlap expected by chance was E = n_ref n_test / T, where n_ref and n_test are the numbers of boundaries in the reference and test segmentations and T is the number of timepoints"
            + (" (with a tolerance window, E was estimated as the mean overlap across permutations)." if tol else "."),
            _test_sentence(res, cfg),
            "The absolute overlap was OA = (O \u2212 E) / (n_test \u2212 E) and the relative overlap, which corrects for unequal numbers of boundaries, was OR = (O \u2212 E) / (min(n_ref, n_test) \u2212 E); both are 0 at chance level and 1 when every boundary is shared.",
            f"Statistical significance was assessed with a permutation test in which the order of the state durations of the test segmentation was shuffled {so.get('n_perm', 1000)} times, preserving the number and lengths of states, and p was the proportion of permutations with an overlap at least as large as the observed one, counting the observed value.",
        ]
        om = cfg.get("overlap_map")
        if om:
            ref = om["reference"]
            ref_s = ("the stimulus boundaries" if ref["kind"] == "stimulus"
                     else "the boundaries of the whole-brain segmentation")
            sents.append(f"For every parcel, the boundaries of its optimal solution were compared with {ref_s}, and p-values were corrected for multiple comparisons across parcels with the Benjamini\u2013Hochberg false discovery rate procedure {{cite:Benjamini1995}} at q < {cfg['overlap_map'].get('alpha', 0.05):g}.")
            if om.get("delays") and len(om["delays"]) > 1:
                dl = om["delays"]
                sents += [
                    f"Because the haemodynamic delay may differ between regions, the stimulus boundaries were shifted by each of {len(dl)} delays from {dl[0]:g} to {dl[-1]:g} s in steps of one TR, and each parcel's best delay was the one with the highest OR.",
                    "To correct for this search, the parcel's p-value compared its best OR with the maximum OR across all delays in each permutation.",
                ]
        d.section("Boundary overlap", sents)

    cl = cfg.get("clusters")
    if cl:
        params = cl["params"]
        method_s = {
            "hierarchical": f"into {params.get('n_clusters')} clusters with agglomerative hierarchical clustering, using average linkage on the correlation distance (1 \u2212 r)",
            "dbscan": f"with DBSCAN {{cite:Ester1996}} on the cosine distance, with a neighbourhood radius of {params.get('eps', 0):.3g} and a minimum of {params.get('min_samples')} states per core point",
            "kmeans": f"into {params.get('n_clusters')} clusters with k-means clustering",
        }[cl["method"]]
        d.section("State pattern clustering", [
            f"To identify states with similar activity patterns, the mean pattern of each of the {cl['k']} states of the {'optimal' if cl['k'] == res['segmentation']['best_k'] else 'selected'} solution was z-scored across {unit}s, and the patterns were grouped {method_s}, as implemented in {'SciPy {cite:Virtanen2020}' if cl['method'] == 'hierarchical' else 'scikit-learn {cite:Pedregosa2011}'}.",
            _cluster_choice(cl),
            "Transitions between clusters were counted between consecutive states assigned to different clusters"
            + ("; transitions to or from states that were not assigned to any cluster were not counted." if cl["method"] == "dbscan" else "."),
        ])

    if data["kind"] == "nifti" and len(figs) > 1:
        d.section("Visualisation", [
            "Parcel-level maps were displayed on the inflated fsaverage surface {cite:Fischl2012} by assigning each vertex the value of its parcel in the surface version of the Schaefer atlas {cite:Schaefer2018}; network summaries used the assignment of parcels to the networks of {citet:Yeo2011}.",
        ])

    sw = cfg["package"]["software"]
    d.section("Software and code availability", [
        "Analyses were performed in Python " + sw.get("python", "") + " with NumPy " + sw.get("numpy", "") + ", SciPy " + sw.get("scipy", "") + ", nilearn " + sw.get("nilearn", "") + ", scikit-learn " + sw.get("scikit-learn", "") + " and statesegmentation " + sw.get("statesegmentation", "") + ".",
        "The analysis scripts, which reproduce every number and figure reported here, were generated with GSBS GUI"
        + (f" (commit {cfg['package']['gsbs_gui_commit'][:7]})" if cfg["package"].get("gsbs_gui_commit") else "")
        + " and are available at [TODO: repository or DOI].",
    ])
    return d


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def results(cfg: dict, res: dict) -> Doc:
    figs = figure_order(cfg, res)
    d = Doc(figs)
    seg = res["segmentation"]
    tr = cfg["data"].get("tr")
    target = cfg["segmentation"]["target"]
    what = {"whole_brain": "the whole-brain parcel time series", "data": "the time series"}.get(
        target, f"the voxel time series of parcel {target}")
    s1 = [f"GSBS applied to {what} reached its t-distance optimum at {plural(seg['best_k'], 'state')} (t = {seg['t_best']:.1f}; {{fig:segmentation:A}})."]
    if seg["best_at_limit"]:
        s1.append("This optimum lay at the largest number of states evaluated, so no optimum was found within the search range [TODO: refit with a larger kmax or interpret with caution].")
    lengths_s = (f"{seg['median_length_s']:.3g} s (range {seg['min_length_s']:.3g} to {seg['max_length_s']:.3g} s; {{fig:segmentation:B, C}})"
                 if tr else f"{seg['median_length_trs']:g} TRs (range {seg['min_length_trs']} to {seg['max_length_trs']} TRs; {{fig:segmentation:B, C}})")
    if seg["k"] == seg["best_k"]:
        s1.append(f"The states of this solution had a median duration of {lengths_s}.")
    else:
        s1.append(f"For the analyses below we used the solution with {plural(seg['k'], 'state')} [TODO: justify the choice of this solution], whose states had a median duration of {lengths_s}.")
    d.section(None, s1)

    be = res.get("boundary_example")
    if be:
        d.section(None, [
            f"The strongest boundary of this solution, between states {be['index']} and {be['index'] + 1}, occurred at TR {be['tr']}"
            + (f" ({be['tr'] * tr:.0f} s)" if tr else "") + f", with a boundary strength (1 \u2212 r between the mean patterns of the two states) of {be['strength']:.2f} ({{fig:boundary}}).",
            f"The change in activity across this boundary was largest in the {be['top_network']} network ({{fig:boundary:D}}).",
        ])

    ts = res.get("timescales")
    if ts and len(ts["rois"]) >= 2:
        ns = np.array(ts["nstates"])
        med = np.array(ts.get("median_length_s") or ts["median_length_trs"])
        unit = "s" if ts.get("median_length_s") else "TRs"
        d.section(None, [
            f"Across the {plural(len(ts['rois']), 'parcel')} analysed at the region level, the optimal number of states ranged from {ns.min()} to {ns.max()} (median {np.median(ns):g}), corresponding to median state durations of {med.min():.3g} to {med.max():.3g} {unit} ({{fig:timescales}})."
            + (f" {cap(plural(len(ts['excluded']), 'parcel', words=True))} reached the search limit without an optimum and {'was' if len(ts['excluded']) == 1 else 'were'} excluded." if ts["excluded"] else ""),
        ])

    so = res.get("stimulus_overlap")
    if so:
        st = cfg["stimulus"]
        d.section(None, [
            f"Of the {plural(so['n_test'], 'neural state boundary')}, {so['observed']} {'coincided with' if not so['tolerance'] else 'fell within ' + plural(so['tolerance'], 'TR') + ' of'} one of the {so['n_ref']} {st['label'].lower()} after shifting the latter by {st['delay_s']:g} s, compared with {so['expected']:.1f} expected by chance (OA = {so['oa']:.2f}, OR = {so['or']:.2f}; permutation test, {fmt_p(so['p'], so['n_perm'])}; {{fig:stimulus_overlap}}).",
        ])

    om = res.get("overlap_map")
    if om:
        q = np.array(om["q"])
        n_sig = int(np.sum(q < 0.05))
        orv = np.array(om["or"])
        ref = cfg["overlap_map"]["reference"]
        ref_s = (cfg["stimulus"]["label"].lower() if ref["kind"] == "stimulus" else "whole-brain state boundaries")
        sig_s = ("no parcel showed" if n_sig == 0 else f"{plural(n_sig, 'parcel')} showed")
        sents = [f"Comparing the boundaries of each of the {plural(len(om['rois']), 'parcel')} with the {ref_s}, OR ranged from {orv.min():.2f} to {orv.max():.2f} (median {np.median(orv):.2f}), and {sig_s} overlap above chance after false discovery rate correction (q < 0.05; {{fig:overlap_map:A}})."]
        if om.get("grid"):
            bd = np.array([v for v in om["best_delay"]], float)
            sents.append(f"The delay that maximised the overlap varied between parcels from {np.nanmin(bd):g} to {np.nanmax(bd):g} s (median {np.nanmedian(bd):g} s; {{fig:overlap_map:B}}).")
        nm = om.get("network_means") or {}
        if nm:
            top = max(nm.items(), key=lambda kv: kv[1]["mean"])
            sents.append(f"Averaged within networks over all analysed parcels, overlap was highest in the {top[0]} network (mean OR = {top[1]['mean']:.2f}, {top[1]['n_significant']} of {plural(top[1]['n'], 'parcel')} significant).")
        d.section(None, sents)

    cl = res.get("clusters")
    if cl:
        n = len(cl["centroids"])
        tf = np.array(cl["time_fraction"])
        t = np.array(cl["transitions"])
        noise = int(np.sum(np.array(cl["labels"]) < 0))
        noise_s = (f", with {plural(noise, 'state')} not assigned to any cluster" if noise else "")
        sil = (f"mean silhouette {cl['silhouette']:.2f}" + (", unclustered states counted as 0" if noise else "")
               if cl["silhouette"] is not None and not np.isnan(cl["silhouette"]) else "silhouette undefined")
        sents = [f"Clustering the {cl['k']} state patterns yielded {plural(n, 'cluster')} ({sil}; {{fig:clusters:A}}){noise_s}, and the largest cluster covered {100 * tf[0]:.0f}% of the time course ({{fig:clusters:B}})."]
        total = int(t.sum())
        if total >= 5:
            a, b = np.unravel_index(np.argmax(t), t.shape)
            sents.append(f"Of the {total} transitions between clusters, the most frequent was from cluster {a + 1} to cluster {b + 1} ({plural(int(t[a, b]), 'transition')}; {{fig:clusters:C}}).")
        elif total > 0:
            sents.append(f"There {'was' if total == 1 else 'were'} {plural(total, 'transition')} between clusters"
                         + (", not counting transitions to or from unclustered states" if noise else "")
                         + " ({fig:clusters:C}).")
        d.section(None, sents)
    return d


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def captions(cfg: dict, res: dict) -> dict:
    seg = res["segmentation"]
    tr = cfg["data"].get("tr")
    target = cfg["segmentation"]["target"]
    what = {"whole_brain": "whole-brain parcel", "data": ""}.get(target, f"parcel {target} voxel")
    out = {}
    out["segmentation"] = (
        f"**Neural state segmentation of the {what} time series.** "
        f"**A**, t-distance for every solution evaluated by GSBS; the star marks the optimum ({seg['best_k']} states)"
        + ("" if seg["k"] == seg["best_k"] else f" and the dashed line the solution shown ({seg['k']} states)") + ". "
        f"**B**, correlation between the activity patterns of all pairs of timepoints, with the {seg['k']} states outlined. "
        f"**C**, the z-scored time series with the state boundaries"
        + (" and, above, the stimulus boundaries shifted by the haemodynamic delay (green)" if res.get("stimulus_overlap") else "") + ". "
        "Blocks of high within-state correlation along the diagonal of B correspond to the states.")
    be = res.get("boundary_example")
    if be:
        out["boundary"] = (
            f"**Activity patterns around the strongest state boundary** (TR {be['tr']}"
            + (f", {be['tr'] * tr:.0f} s" if tr else "") + "). "
            f"**A, B**, mean z-scored activity of the states before and after the boundary on the inflated cortical surface. "
            "**C**, the difference between them. **D**, the difference averaged within each network. "
            f"The boundary strength (1 \u2212 r between the two patterns) was {be['strength']:.2f}.")
    ts = res.get("timescales")
    if ts and len(ts["rois"]) >= 2:
        out["timescales"] = (
            f"**Optimal number of states per parcel.** Number of states at the t-distance optimum for each of the {len(ts['rois'])} parcels analysed, with network means on the right. "
            "Grey parcels were not analysed" + (" or reached the search limit without an optimum" if ts["excluded"] else "") + ".")
    so = res.get("stimulus_overlap")
    if so:
        st = cfg["stimulus"]
        out["stimulus_overlap"] = (
            f"**Overlap between neural state boundaries and {st['label'].lower()}.** "
            f"**A**, {st['label'].lower()} shifted by {st['delay_s']:g} s (top) and neural state boundaries (bottom); neural boundaries {'at the same TR as' if not so['tolerance'] else 'within ' + plural(so['tolerance'], 'TR') + ' of'} a stimulus boundary are highlighted in red. "
            f"**B**, the number of shared boundaries after shuffling the order of the neural state durations ({so['n_perm']} permutations), with the observed value (red) and the value expected by chance (dashed). "
            f"OA = {so['oa']:.2f}, OR = {so['or']:.2f}, {fmt_p(so['p'], so['n_perm'])}.")
    om = res.get("overlap_map")
    if om:
        ref = cfg["overlap_map"]["reference"]
        ref_s = (cfg["stimulus"]["label"].lower() if ref["kind"] == "stimulus" else "whole-brain state boundaries")
        out["overlap_map"] = (
            f"**Overlap of each parcel's state boundaries with the {ref_s}.** "
            f"**A**, relative overlap (OR) for parcels with q < 0.05 after false discovery rate correction across {len(om['rois'])} parcels; other analysed parcels and parcels not analysed are grey. "
            + ("**B**, the haemodynamic delay giving each parcel's highest overlap, for all analysed parcels; p-values in A are corrected for this search over delays. " if om.get("grid") else "")
            + "Bars on the right show network means over all analysed parcels, with the number of significant and analysed parcels per network.")
    cl = res.get("clusters")
    if cl:
        out["clusters"] = (
            f"**Clusters of state activity patterns.** **A**, mean z-scored pattern of each cluster of the {cl['k']} states, with the number of states and the percentage of the time course it covers. "
            "**B**, the cluster of every state over the time course and the share of the time course in each cluster" + ("; states not assigned to any cluster are grey" if np.any(np.array(cl["labels"]) < 0) else "") + ". "
            "**C**, the probability of moving to each other cluster at a state boundary, with the number of transitions in each cell.")
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_all(cfg: dict, res: dict, out_dir) -> dict:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m, r = methods(cfg, res), results(cfg, res)
    caps = captions(cfg, res)
    figs = figure_order(cfg, res)
    (out / "methods.md").write_text(m.markdown("Methods") + "\n")
    (out / "results.md").write_text(r.markdown("Results") + "\n" + _figure_list_md(figs, caps))
    (out / "methods.tex").write_text(m.latex("Methods"))
    (out / "results.tex").write_text(r.latex("Results"))
    (out / "figures.tex").write_text(_figures_tex(figs, caps))
    cited = list(dict.fromkeys(m.cited + r.cited))
    (out / "references.bib").write_text("\n\n".join(REFS[k][1] for k in cited) + "\n")
    return {"figures": figs, "captions": caps}


def _figure_list_md(figs, caps) -> str:
    lines = ["## Figure captions", ""]
    for i, f in enumerate(figs, 1):
        lines += [f"**Figure {i}.** {caps[f]}", ""]
    return "\n".join(lines)


def _figures_tex(figs, caps) -> str:
    out = []
    for f in figs:
        cap = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", caps[f]).replace("%", "\\%")
        cap = cap.replace("\u2212", "$-$")
        out += ["\\begin{figure}[t]", "\\centering",
                f"\\includegraphics[width=\\textwidth]{{figures/{f}.pdf}}",
                f"\\caption{{{cap}}}", f"\\label{{fig:{f}}}", "\\end{figure}", ""]
    return "\n".join(out)
