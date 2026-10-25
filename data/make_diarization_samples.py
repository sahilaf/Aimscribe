"""Task 1 deliverable "sample audio with reference and predicted speaker annotations".

One self-contained folder per sample under data/samples/diarization/<sample>/:
  audio.wav                 16 kHz mono (DiarBench and synthetic clips are copied from DATA_ROOT)
  reference.rttm            human / generated reference ("who spoke when")
  reference_overlap.rttm    regions where 2+ reference speakers talk at once
  predicted_<system>.rttm   system output BEFORE and AFTER the improvement (copied unchanged from results/)
  timeline.png              reference vs each system; predicted speakers coloured by their best-matching
                            reference speaker (the DER mapping), red = reference overlap
plus data/samples/diarization/README.md and samples.csv with the DER of every system on every sample.

The predictions are the saved outputs of the reported runs, not re-runs; DER is recomputed here with the
same metric (pyannote.metrics, collar 0.25 s) and matches results/task1 + results/task3.

Usage: python data/make_diarization_samples.py
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pyannote.core import Annotation, Segment, Timeline  # noqa: E402
from pyannote.database.util import load_rttm  # noqa: E402
from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_ROOT, REPO  # noqa: E402
from prepare_diarbench import overlap_regions  # noqa: E402
from prepare_diarization import write_rttm  # noqa: E402

OUT = REPO / "data" / "samples" / "diarization"
R1, R3 = REPO / "results" / "task1", REPO / "results" / "task3"
DB = REPO / "data" / "diarbench" / "test"

BASE31, C1 = "pyannote31", "community1"
# (sample, source set, audio, reference rttm, reference overlap rttm or None, {system: predicted rttm}, what to look at)
SAMPLES = [
    ("diarbench_bengali_012", "DiarBench test (near-field, 4 speakers)", DATA_ROOT / "diarbench/test/bengali_012.wav",
     DB / "rttm/bengali_012.rttm", DB / "overlap/bengali_012.rttm",
     {BASE31: R1 / "db_test_baseline/rttm/bengali_012.rttm", C1: R1 / "db_test_community1/rttm/bengali_012.rttm"},
     "one speaker identified as multiple speakers: 3.1 finds 5 speakers for 4; community-1 finds 4"),
    ("diarbench_bengali_097", "DiarBench test (near-field, 2 speakers, 13 % overlap)", DATA_ROOT / "diarbench/test/bengali_097.wav",
     DB / "rttm/bengali_097.rttm", DB / "overlap/bengali_097.rttm",
     {BASE31: R1 / "db_test_baseline/rttm/bengali_097.rttm", C1: R1 / "db_test_community1/rttm/bengali_097.rttm"},
     "overlapping speech (13 %): DER inside overlap 7 % for both; community-1 removes most confusion (2.8 -> 0.5)"),
    ("diarbench_bengali_072", "DiarBench test (far-field, 2 speakers)", DATA_ROOT / "diarbench/test/bengali_072.wav",
     DB / "rttm/bengali_072.rttm", DB / "overlap/bengali_072.rttm",
     {BASE31: R1 / "db_test_baseline/rttm/bengali_072.rttm", C1: R1 / "db_test_community1/rttm/bengali_072.rttm"},
     "incorrect speaker assignment in far-field audio: confusion 31.8 -> 14.0; overlap stays the hardest part (54 %)"),
    ("diarbench_bengali_013", "DiarBench test (near-field, 4 speakers, 12 % overlap)", DATA_ROOT / "diarbench/test/bengali_013.wav",
     DB / "rttm/bengali_013.rttm", DB / "overlap/bengali_013.rttm",
     {BASE31: R1 / "db_test_baseline/rttm/bengali_013.rttm", C1: R1 / "db_test_community1/rttm/bengali_013.rttm"},
     "a clip where the improvement does NOT help (kept on purpose): both systems find 3 of 4 speakers"),
    ("sim_bn_conv_00", "simulated conversation (OpenSLR speakers, 2 speakers)", DATA_ROOT / "diarization/wav/bn_conv_00.wav",
     REPO / "data/diarization/rttm/bn_conv_00.rttm", None,
     {BASE31: R1 / "baseline/rttm/bn_conv_00.rttm", "pyannote31_recluster": R1 / "rc_t0.70/rttm/bn_conv_00.rttm"},
     "incorrect speaker assignment (15.4-17.3 s given to the other speaker); fixed by my segment re-clustering"),
    ("sim_bn_conv_09", "simulated conversation (OpenSLR speakers, 3 speakers, 15 dB noise)", DATA_ROOT / "diarization/wav/bn_conv_09.wav",
     REPO / "data/diarization/rttm/bn_conv_09.rttm", None,
     {BASE31: R1 / "baseline/rttm/bn_conv_09.rttm", "pyannote31_recluster": R1 / "rc_t0.70/rttm/bn_conv_09.rttm"},
     "speaker merge (3 -> 2) by 3.1; re-clustering separates them but over-splits (4)"),
    ("real_consult_A", "my real role-play consultation (doctor + patient)", REPO / "data/consultation/audio/consult_A.wav",
     REPO / "data/consultation/rttm/consult_A.rttm", None,
     {BASE31: R3 / "baseline/consult_A.rttm", C1: R3 / "final_c1/consult_A.rttm"},
     "clean 2-speaker recording: both systems near-perfect (DER 1.6)"),
    ("real_consult_B", "my real role-play consultation (attendant + doctor, attendant speaks first)",
     REPO / "data/consultation/audio/consult_B.wav", REPO / "data/consultation/rttm/consult_B.rttm", None,
     {BASE31: R3 / "baseline/consult_B.rttm", C1: R3 / "final_c1/consult_B.rttm"},
     "one speaker identified as multiple: 3.1 gives the attendant's short replies a third label; community-1 finds 2"),
    ("synthetic_S05", "synthetic consultation (Parler-TTS, doctor interrupts patient)", DATA_ROOT / "tts_consultation/S05.wav",
     REPO / "data/tts_consultation/rttm/S05.rttm", None,
     {BASE31: R3 / "tts_baseline/S05.rttm", C1: R3 / "tts_final_c1/S05.rttm"},
     "annotated overlap (doctor interrupts at 5.27 s, patient speaks until 5.77 s); all errors are missed speech, mostly the overlapped second voice"),
]


def load(path, uri):
    return load_rttm(path).get(uri) or Annotation(uri=uri)


def plot(name, ref, ov, systems, scores, duration, path):
    """Reference lanes on top, one panel per system below; predicted labels mapped to reference speakers."""
    ref_spk = sorted(ref.labels())
    colour = {s: f"C{i}" for i, s in enumerate(ref_spk)}
    panels = [("reference", ref, {s: s for s in ref_spk})]
    for sys_name, hyp in systems.items():
        mapping = DiarizationErrorRate().optimal_mapping(ref, hyp)        # hyp label -> ref label
        panels.append((f"{sys_name}  (DER {scores[sys_name]:.1f} %)", hyp, mapping))
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 1.2 + 0.9 * sum(max(1, len(p[1].labels())) for p in panels) / 2),
                             sharex=True, squeeze=False)
    for ax, (title, ann, mapping) in zip(axes[:, 0], panels):
        labels = sorted(ann.labels(), key=lambda l: (mapping.get(l) is None, str(mapping.get(l, l))))
        for seg, _, lab in ann.itertracks(yield_label=True):
            ref_lab = mapping.get(lab)
            ax.barh(labels.index(lab), seg.duration, left=seg.start, height=0.7,
                    color=colour.get(ref_lab, "0.45"), edgecolor="none")
        for seg in ov:
            ax.axvspan(seg.start, seg.end, color="red", alpha=0.18, lw=0)
        ax.set_yticks(range(len(labels)),
                      [f"{l} → {mapping[l]}" if mapping.get(l) and mapping[l] != l else str(l) for l in labels], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.invert_yaxis()
    axes[-1, 0].set_xlim(0, duration)
    axes[-1, 0].set_xlabel("time (s)   colour = reference speaker (grey = extra speaker with no match)   red = reference overlap")
    fig.suptitle(name, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    import soundfile as sf
    for old in OUT.glob("*.wav"):            # earlier flat layout (bn_conv_00/09.wav) -> now inside sim_* folders
        old.unlink()
    rows = []
    for name, source, audio, ref_path, ov_path, systems, note in SAMPLES:
        uri = ref_path.stem
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(audio, d / "audio.wav")
        duration = sf.info(d / "audio.wav").duration
        shutil.copy(ref_path, d / "reference.rttm")
        ref = load(ref_path, uri)
        if ov_path:
            shutil.copy(ov_path, d / "reference_overlap.rttm")
        else:
            segs = [{"speaker": l, "start": s.start, "end": s.end} for s, _, l in ref.itertracks(yield_label=True)]
            write_rttm([{"speaker": "overlap", "start": a, "end": b} for a, b in overlap_regions(segs)], uri,
                       d / "reference_overlap.rttm")
        ov = load(d / "reference_overlap.rttm", uri).get_timeline().support()
        hyps, scores = {}, {}
        for sys_name, p in systems.items():
            shutil.copy(p, d / f"predicted_{sys_name}.rttm")
            hyp = load(p, uri)
            hyps[sys_name] = hyp
            det = DiarizationErrorRate(collar=0.25)(ref, hyp, detailed=True)
            tot = det["total"]
            der_ov = 100 * DiarizationErrorRate(collar=0.25)(ref, hyp, uem=ov) if ov else None
            scores[sys_name] = 100 * det["diarization error rate"]
            rows.append({"sample": name, "set": source, "duration_s": round(duration, 1),
                         "ref_speakers": len(ref.labels()), "overlap_%": round(100 * ov.duration() / tot, 1),
                         "system": sys_name, "found_speakers": len(hyp.labels()),
                         "DER%": round(scores[sys_name], 2), "miss%": round(100 * det["missed detection"] / tot, 2),
                         "FA%": round(100 * det["false alarm"] / tot, 2), "conf%": round(100 * det["confusion"] / tot, 2),
                         "DER_in_overlap%": None if der_ov is None else round(der_ov, 1), "what_to_look_at": note})
        plot(f"{name}: {source}", ref, ov, hyps, scores, duration, d / "timeline.png")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "samples.csv", index=False, encoding="utf-8")

    lines = ["# Diarization samples: audio + reference + predicted speaker annotations", "",
             "Each folder: `audio.wav`, `reference.rttm`, `reference_overlap.rttm`, `predicted_<system>.rttm` (before and "
             "after the improvement, the saved outputs of the reported runs) and `timeline.png`. Built by "
             "`data/make_diarization_samples.py`; DER with pyannote.metrics, collar 0.25 s. RTTM: "
             "`SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>`; overlapping speech = two lines "
             "covering the same time.", "",
             "| sample | source | dur (s) | ref spk | overlap % | system | spk found | DER % | miss | FA | conf | DER in overlap | what to look at |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| `{r.sample}` | {r.set} | {r.duration_s} | {r.ref_speakers} | {r._5} | {r.system} | "
                     f"{r.found_speakers} | **{r._8}** | {r._9} | {r._10} | {r._11} | "
                     f"{'' if pd.isna(r._12) else r._12} | {r.what_to_look_at} |")
    lines += ["", "Systems: `pyannote31` = pyannote/speaker-diarization-3.1 (baseline); `community1` = "
              "pyannote/speaker-diarization-community-1 (the Task 1 improvement chosen on DiarBench dev, used in the final "
              "pipeline); `pyannote31_recluster` = 3.1 + my segment-level re-clustering (the improvement on simulated "
              "conversations). DiarBench audio: sarvamai/indic-diarbench (CC BY 4.0)."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df[["sample", "system", "ref_speakers", "found_speakers", "DER%", "miss%", "FA%", "conf%", "DER_in_overlap%"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
