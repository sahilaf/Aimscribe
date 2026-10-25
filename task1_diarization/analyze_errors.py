"""Task 1: find and categorise real diarization errors, utterance by utterance.

1. Map system labels (SPEAKER_00...) to reference speakers with the same optimal
   one-to-one mapping DER uses. System clusters left unmapped are "extra" speakers.
2. On a 10 ms grid, check every reference utterance: what fraction of its speech
   was given to the right speaker, missed, or given to someone else.
3. Put each badly handled utterance (>= 50% wrong) into the assessment's categories:
     missed_short      short utterance (< 1.6 s) mostly not detected
     overlap           error happens where 2+ people talk at once
     speaker_split     given to an EXTRA cluster: one person became two "speakers"
     speaker_merge     the person has no cluster at all: merged into someone else
     wrong_speaker     given to another real speaker's cluster
     missed            (longer) speech not detected
4. Draw reference vs. system timelines with an error strip.

Usage: python task1_diarization/analyze_errors.py --run baseline
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import REPO  # noqa: E402

STEP = 0.01      # 10 ms frames
SHORT = 1.6      # seconds
BAD = 0.5        # an utterance counts as an error example if >= 50% of its speech is wrong


def to_frames(annotation, n):
    """Annotation -> list of label sets, one per 10 ms frame."""
    frames = [set() for _ in range(n)]
    for seg, _, label in annotation.itertracks(yield_label=True):
        for i in range(int(seg.start / STEP), min(n, int(seg.end / STEP))):
            frames[i].add(label)
    return frames


def analyse_clip(uri, ref, hyp, utterances):
    mapping = DiarizationErrorRate().optimal_mapping(ref, hyp)   # {hyp_label: ref_label}
    hyp = hyp.rename_labels(mapping)
    extra = set(hyp.labels()) - set(ref.labels())                  # clusters with no real speaker
    unmapped_ref = set(ref.labels()) - set(mapping.values())       # real speakers with no cluster
    n = int(max(ref.get_timeline().extent().end, hyp.get_timeline().extent().end if hyp else 0) / STEP) + 1
    R, H = to_frames(ref, n), to_frames(hyp, n)

    rows = []
    for u in utterances:
        spk, a, b = u["speaker"], int(u["start"] / STEP), min(n, int(u["end"] / STEP))
        idx = [i for i in range(a, b) if spk in R[i]]               # this speaker's speech frames
        if not idx:
            continue
        correct = sum(spk in H[i] for i in idx)
        missed = sum(not H[i] for i in idx)
        wrong = len(idx) - correct - missed
        overlap = [i for i in idx if len(R[i]) >= 2]
        err_frames = [i for i in idx if spk not in H[i]]
        err_in_overlap = sum(i in set(overlap) for i in err_frames)
        given_to = Counter(l for i in err_frames for l in H[i]).most_common(1)
        given_to = given_to[0][0] if given_to else None
        dur = len(idx) * STEP
        err = 1 - correct / len(idx)

        if err < BAD:
            category = "ok"
        elif err_frames and err_in_overlap / len(err_frames) >= 0.5:
            category = "overlap"
        elif missed >= wrong:
            category = "missed_short" if dur < SHORT else "missed"
        elif spk in unmapped_ref:
            category = "speaker_merge"
        elif given_to in extra:
            category = "speaker_split"
        else:
            category = "wrong_speaker"
        rows.append({"clip": uri, "speaker": spk, "start": u["start"], "end": u["end"],
                     "speech_s": round(dur, 2), "short": dur < SHORT, "overlap_frac": round(len(overlap) / len(idx), 2),
                     "error_frac": round(err, 2), "missed_frac": round(missed / len(idx), 2),
                     "given_to": given_to, "category": category, "text": u["text"]})
    return rows, hyp, extra, R, H, n


def plot_clip(uri, ref, hyp, extra, R, H, n, path):
    speakers = sorted(ref.labels()) + sorted(extra)
    colors = {s: f"C{i}" for i, s in enumerate(speakers)}
    fig, axes = plt.subplots(3, 1, figsize=(16, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [len(ref.labels()), max(1, len(hyp.labels())), 0.5]})
    for ax, ann, title in [(axes[0], ref, "reference"), (axes[1], hyp, "system (mapped)")]:
        labels = sorted(ann.labels(), key=speakers.index)
        for seg, _, l in ann.itertracks(yield_label=True):
            ax.barh(labels.index(l), seg.duration, left=seg.start, height=0.7, color=colors[l],
                    hatch="//" if l in extra else None, edgecolor="k" if l in extra else None, linewidth=0.3)
        ax.set_yticks(range(len(labels)), [f"{l} (EXTRA)" if l in extra else l for l in labels], fontsize=8)
        ax.set_ylabel(title, fontsize=8)
    # error strip: confusion red, miss orange, false alarm purple
    t = np.arange(n) * STEP
    conf = np.array([min(len(r), len(h)) - len(r & h) > 0 for r, h in zip(R, H)])
    miss = np.array([len(r) > len(h) for r, h in zip(R, H)])
    fa = np.array([len(h) > len(r) for r, h in zip(R, H)])
    for mask, c, lab in [(conf, "red", "confusion"), (miss, "orange", "missed"), (fa, "purple", "false alarm")]:
        axes[2].fill_between(t, 0, 1, where=mask, color=c, step="post", label=lab)
    axes[2].set_yticks([])
    axes[2].legend(loc="upper right", ncol=3, fontsize=7)
    axes[2].set_xlabel("time (s)")
    axes[0].set_title(uri)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run_dir = REPO / "results" / "task1" / args.run
    (run_dir / "plots").mkdir(exist_ok=True)

    all_rows = []
    for ref_path in sorted((REPO / "data" / "diarization" / "rttm").glob("*.rttm")):
        uri = ref_path.stem
        ref = load_rttm(ref_path)[uri]
        hyp = load_rttm(run_dir / "rttm" / f"{uri}.rttm")[uri]
        utts = json.loads((REPO / "data" / "diarization" / "transcripts" / f"{uri}.json").read_text(encoding="utf-8"))
        rows, hyp_m, extra, R, H, n = analyse_clip(uri, ref, hyp, utts)
        all_rows += rows
        plot_clip(uri, ref, hyp_m, extra, R, H, n, run_dir / "plots" / f"{uri}.png")

    df = pd.DataFrame(all_rows)
    df.to_csv(run_dir / "utterance_errors.csv", index=False, encoding="utf-8")

    print(f"\n=== {args.run}: utterances by category (of {len(df)}) ===")
    print(df["category"].value_counts().to_string())
    print("\nShort utterances (<1.6 s): "
          f"{(df[df.short].error_frac >= BAD).mean() * 100:.0f}% handled badly vs "
          f"{(df[~df.short].error_frac >= BAD).mean() * 100:.0f}% of longer ones")
    ov = df.overlap_frac > 0
    print(f"Utterances touching overlap: {(df[ov].error_frac >= BAD).mean() * 100:.0f}% handled badly vs "
          f"{(df[~ov].error_frac >= BAD).mean() * 100:.0f}% without overlap")
    print("\nWorst example per category:")
    worst = df[df.category != "ok"].sort_values("error_frac", ascending=False).groupby("category").head(1)
    print(worst[["category", "clip", "speaker", "start", "end", "error_frac", "given_to", "text"]].to_string(index=False))


if __name__ == "__main__":
    main()
