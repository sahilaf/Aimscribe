"""Quality check of the synthetic training clips (data/make_tts_train.py --synth) before fine-tuning.

IndicF5 sometimes produces broken audio (wrong words, mumbling). Each clip is transcribed with IndicConformer 600M;
clips whose CER against their own text is above 50 % are dropped. Writes:
  data/tts_train/qc_600m.csv        id, ref, hyp, cer for every synthesised clip
  data/tts_train/manifest_tts.jsonl NeMo manifest of the kept clips (the fine-tuning input)
Reported run: 1,058 clips synthesised, 46 dropped, 1,012 kept (81.4 min), 600M CER 6.6 % on the kept clips.

Usage: python data/qc_tts_train.py --model <DATA_ROOT>/models/indic-conformer-600m
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task2_asr"))
from common import DATA_ROOT, REPO, normalize_bn  # noqa: E402

OUT = REPO / "data" / "tts_train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA_ROOT / "models" / "indic-conformer-600m"))
    ap.add_argument("--max_cer", type=float, default=0.5)
    args = ap.parse_args()
    from other_asr import IndicConformerASR

    wav_dir = DATA_ROOT / "tts_train"
    df = pd.read_csv(OUT / "train_sentences.tsv", sep="\t")
    df = df[df.id.map(lambda i: (wav_dir / f"{i}.wav").exists())]
    asr = IndicConformerASR(args.model)
    hyp = asr.transcribe([sf.read(wav_dir / f"{i}.wav", dtype="float32")[0] for i in df.id])
    df["ref"] = df.text.map(normalize_bn)
    df["hyp"] = [normalize_bn(h) for h in hyp]
    df["cer"] = [jiwer.cer(r, h) for r, h in zip(df.ref, df.hyp)]
    df[["id", "ref", "hyp", "cer"]].to_csv(OUT / "qc_600m.csv", index=False)
    keep = df[df.cer <= args.max_cer]
    with open(OUT / "manifest_tts.jsonl", "w", encoding="utf-8") as f:
        for r in keep.itertuples():
            p = wav_dir / f"{r.id}.wav"
            f.write(json.dumps({"audio_filepath": str(p), "duration": round(sf.info(p).duration, 2), "text": r.ref,
                                "lang": "bn"}, ensure_ascii=False) + "\n")
    minutes = sum(sf.info(wav_dir / f"{i}.wav").duration for i in keep.id) / 60
    print(f"{len(df)} clips, kept {len(keep)} (dropped {len(df) - len(keep)} with CER > {args.max_cer:.0%}), "
          f"{minutes:.1f} min | 600M CER on kept {100 * jiwer.cer(list(keep.ref), list(keep.hyp)):.1f}%")


if __name__ == "__main__":
    main()
