"""Task 2, Phase 1: build a small Bengali ASR train/test set from OpenSLR 53.

Steps
  1. Read transcripts and extract the audio zip.
  2. Clean: drop clips that are too short/long or whose text has digits or
     Latin letters (ambiguous to transcribe: "৫" could be read as "পাঁচ").
  3. Speaker-disjoint split: no test speaker is ever seen in training, so the
     test WER measures generalisation to new voices, not memorisation.
  4. Preprocess audio: mono, 16 kHz, trim long leading/trailing silence
     (keeping a safety pad so we never cut a word), loudness-normalise.
  5. Write train.csv / test.csv and copy a few samples into the repo.

Usage:  python data/prepare_asr.py --train_minutes 90 --test_minutes 20
"""
import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from common import DATA_ROOT, RAW, REPO, SR, extract_zip, normalize_bn, read_transcripts

MIN_DUR, MAX_DUR = 1.0, 20.0     # Whisper handles <=30 s; very short clips carry little signal
TRIM_TOP_DB = 35                 # anything 35 dB below the peak counts as silence
KEEP_PAD = 0.15                  # seconds of silence kept around speech so first/last words survive
TARGET_RMS_DB = -20.0
_BAD_CHARS = re.compile(r"[0-9০-৯A-Za-z]")


def preprocess(in_path, out_path):
    audio, sr = sf.read(in_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    _, (start, end) = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    pad = int(KEEP_PAD * SR)
    audio = audio[max(0, start - pad): min(len(audio), end + pad)]
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    audio = audio * (10 ** (TARGET_RMS_DB / 20) / rms)
    audio = np.clip(audio, -0.99, 0.99)
    sf.write(out_path, audio, SR, subtype="PCM_16")
    return len(audio) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(RAW / "asr_bengali_0.zip"))
    ap.add_argument("--tsv", default=str(RAW / "utt_spk_text.tsv"))
    ap.add_argument("--train_minutes", type=float, default=90)
    ap.add_argument("--test_minutes", type=float, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    audio_index = extract_zip(Path(args.zip), RAW / "extracted")
    rows = [r for r in read_transcripts(Path(args.tsv)) if r["utt"] in audio_index]
    print(f"Utterances with audio: {len(rows)}")

    stats = defaultdict(int)
    by_speaker = defaultdict(list)
    for r in rows:
        text = normalize_bn(r["text"])
        if not text or _BAD_CHARS.search(text):
            stats["dropped_text"] += 1
            continue
        dur = sf.info(audio_index[r["utt"]]).duration
        if not MIN_DUR <= dur <= MAX_DUR:
            stats["dropped_duration"] += 1
            continue
        by_speaker[r["speaker"]].append({**r, "text": text, "src": audio_index[r["utt"]], "raw_dur": dur})

    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    split = {"test": [], "train": []}
    budget = {"test": args.test_minutes * 60, "train": args.train_minutes * 60}
    used = {"test": 0.0, "train": 0.0}
    for spk in speakers:  # fill test first, then train, whole speakers at a time
        name = "test" if used["test"] < budget["test"] else "train"
        if used[name] >= budget[name]:
            break
        split[name].extend(by_speaker[spk])
        used[name] += sum(u["raw_dur"] for u in by_speaker[spk])

    for name, items in split.items():
        out_dir = DATA_ROOT / "asr" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for u in items:
            out = out_dir / f"{u['utt']}.wav"
            dur = preprocess(u["src"], out)
            records.append({"path": out.relative_to(DATA_ROOT).as_posix(), "text": u["text"],
                            "speaker": u["speaker"], "duration": round(dur, 2)})
        df = pd.DataFrame(records)
        df.to_csv(REPO / "data" / "asr" / f"{name}.csv", index=False, encoding="utf-8")
        stats[f"{name}_utts"] = len(df)
        stats[f"{name}_speakers"] = df["speaker"].nunique()
        stats[f"{name}_minutes"] = round(df["duration"].sum() / 60, 1)

    # A few samples inside the repo so reviewers can listen without downloading anything
    sample_dir = REPO / "data" / "samples" / "asr"
    sample_dir.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(REPO / "data" / "asr" / "test.csv")
    samples = test_df.sample(10, random_state=args.seed)
    for p in samples["path"]:
        shutil.copy(DATA_ROOT / p, sample_dir)
    samples.assign(path=samples["path"].map(lambda p: Path(p).name)).to_csv(
        sample_dir / "transcripts.csv", index=False, encoding="utf-8")

    overlap = {u["speaker"] for u in split["train"]} & {u["speaker"] for u in split["test"]}
    stats["speaker_overlap_train_test"] = len(overlap)  # must be 0
    print(json.dumps(stats, indent=2))
    (REPO / "data" / "asr" / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
