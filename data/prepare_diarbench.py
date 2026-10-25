"""Real multi-speaker Bengali evaluation set from Indic DiarBench (sarvamai/indic-diarbench, CC BY 4.0).

The benchmark ships audio + a list of human-corrected segments per clip (speaker_id, start_time, end_time,
transcript). This script turns the Bengali part into the same layout as our other eval sets:

  data/diarbench/<split>/rttm/<clip>.rttm         reference "who spoke when" (one line per segment)
  data/diarbench/<split>/transcripts/<clip>.json  one entry per segment: speaker, start, end, text (Task 3)
  data/diarbench/<split>/overlap/<clip>.rttm      regions where 2+ speakers talk at once (overlap-only scoring)
  data/diarbench/split.csv                        clip -> dev / test, split BY RECORDING (clips of one recording
                                                  share speakers and room, so they must not be on both sides)
  data/diarbench/summary.csv                      per clip: condition, speakers, duration, speech, overlap %
Audio goes to DATA_ROOT/diarbench/<split>/<clip>.wav (16 kHz mono). dev = tuning only; test = scored once.

Label checks (problems are counted and reported, never silently "fixed" beyond the two rules below):
  - segments with end <= start are dropped; segments running past the audio end are clipped
  - empty transcripts are kept for diarization (speech is speech) but flagged

Usage (Colab):  python data/prepare_diarbench.py --dev_fraction 0.4
"""
import argparse
import json
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf

from common import DATA_ROOT, REPO, SR
from prepare_diarization import overlap_seconds, plot_timeline, write_rttm

OUT = REPO / "data" / "diarbench"


def to_seconds(t):
    """start/end come as seconds or 'HH:MM:SS(.fff)' strings."""
    if isinstance(t, str) and ":" in t:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(t)


def overlap_regions(segments):
    """[(start, end)] where 2+ speakers are active (sweep over start/end events)."""
    events = sorted([(s["start"], 1) for s in segments] + [(s["end"], -1) for s in segments])
    regions, active, last = [], 0, 0.0
    for t, d in events:
        if active >= 2 and t > last:
            if regions and abs(regions[-1][1] - last) < 1e-6:
                regions[-1] = (regions[-1][0], t)
            else:
                regions.append((last, t))
        active += d
        last = t
    return regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="Bengali")
    ap.add_argument("--dev_fraction", type=float, default=0.4, help="share of RECORDINGS used for tuning")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    from datasets import Audio, load_dataset
    ds = load_dataset("sarvamai/indic-diarbench", args.language, split="test").cast_column("audio", Audio(sampling_rate=SR))

    # split by recording, stratified by condition, so dev and test share no speakers or rooms
    meta = ds.select_columns(["recording_id", "dataset_type"]).to_pandas()
    rng = random.Random(args.seed)
    split = {}
    for cond, g in meta.groupby("dataset_type"):
        recs = sorted(g.recording_id.unique())
        rng.shuffle(recs)
        n_dev = max(1, round(args.dev_fraction * len(recs)))
        split.update({r: ("dev" if i < n_dev else "test") for i, r in enumerate(recs)})

    for part in ["dev", "test"]:
        for sub in ["rttm", "transcripts", "overlap", "plots"]:
            (OUT / part / sub).mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / "diarbench" / part).mkdir(parents=True, exist_ok=True)
    issues, rows = Counter(), []
    for ex in ds:
        clip = str(ex["sample_id"])
        part = split[ex["recording_id"]]
        out, wav_dir = OUT / part, DATA_ROOT / "diarbench" / part
        audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
        dur = len(audio) / SR
        segs = []
        for s in ex["annotated_transcript"]:
            a, b = to_seconds(s["start_time"]), min(to_seconds(s["end_time"]), dur)
            if b <= a:
                issues["dropped: end <= start"] += 1
                continue
            if to_seconds(s["end_time"]) > dur + 0.01:
                issues["clipped: past audio end"] += 1
            text = (s.get("transcript") or "").strip()
            if not text:
                issues["empty transcript (kept)"] += 1
            # speaker ids are free text ("Speaker 0", "New speaker A", "SPEAKER_06"); RTTM fields are space-separated
            spk = re.sub(r"\W+", "_", str(s["speaker_id"])).strip("_") or "unknown"
            segs.append({"speaker": spk, "start": round(a, 3), "end": round(b, 3), "text": text})
        segs.sort(key=lambda s: s["start"])
        sf.write(wav_dir / f"{clip}.wav", audio, SR, subtype="PCM_16")
        write_rttm(segs, clip, out / "rttm" / f"{clip}.rttm")
        ov = overlap_regions(segs)
        write_rttm([{"speaker": "overlap", "start": a, "end": b} for a, b in ov], clip, out / "overlap" / f"{clip}.rttm")
        (out / "transcripts" / f"{clip}.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
        plot_timeline(segs, clip, dur, out / "plots" / f"{clip}.png")
        speech = sum(e - s for s, e in [(x["start"], x["end"]) for x in segs])   # per-speaker sum
        rows.append({"clip": clip, "recording": ex["recording_id"], "condition": ex["dataset_type"], "split": part,
                     "speakers": len({s["speaker"] for s in segs}), "segments": len(segs), "duration_s": round(dur, 1),
                     "speaker_time_s": round(speech, 1), "overlap_s": round(overlap_seconds(segs), 1),
                     "overlap_pct": round(100 * overlap_seconds(segs) / max(speech, 1e-9), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    df[["clip", "recording", "condition", "split"]].to_csv(OUT / "split.csv", index=False)
    print(df.groupby(["split", "condition"]).agg(clips=("clip", "size"), recordings=("recording", "nunique"),
                                                 hours=("duration_s", lambda x: round(x.sum() / 3600, 2)),
                                                 spk_mean=("speakers", "mean"), overlap_pct=("overlap_pct", "mean")).round(1))
    print("label issues:", dict(issues) or "none")


if __name__ == "__main__":
    main()
