"""Task 2: transcribe a test set with a Whisper model and score WER / CER.

Used unchanged for the baseline, the fine-tuned model and every improvement,
so the only thing that differs between runs is the model or decoding setting.

WER = (substitutions + deletions + insertions) / reference words
CER = the same on characters. Bengali words are long and inflected
      (করেছিলেন vs করেছেন), so one wrong suffix costs a whole word in WER but only
      a few characters in CER. CER shows "almost right" output that WER hides.

Both reference and prediction go through the same normalize_bn() (NFC,
punctuation removed), so punctuation and Unicode variants are not counted as errors.

Usage:
  python task2_asr/evaluate_asr.py --model openai/whisper-small --run baseline_small
  python task2_asr/evaluate_asr.py --model /path/to/checkpoint --run finetuned
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import jiwer
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from asr import fix_generation_config  # noqa: E402
from common import DATA_ROOT, REPO, normalize_bn  # noqa: E402

BENGALI = re.compile(r"[ঀ-৿]")


def script_flags(text):
    """Classify output: empty / wrong script (e.g. Kannada, Latin) / bengali."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "empty"
    bn = sum(bool(BENGALI.match(c)) for c in letters)
    return "bengali" if bn / len(letters) >= 0.8 else "wrong_script"


def repetition(text, n=3):
    """Hallucination-loop detector: share of repeated word n-grams."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv", default=str(REPO / "data" / "asr" / "test.csv"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (quick test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model = fix_generation_config(model).to(device).eval()

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    gen = {"language": "bn", "task": "transcribe", "max_new_tokens": args.max_new_tokens,
           "num_beams": args.num_beams}
    if args.no_repeat_ngram_size:
        gen["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    preds, t0 = [], time.time()
    for i in range(0, len(df), args.batch):
        rows = df.iloc[i:i + args.batch]
        audio = [sf.read(DATA_ROOT / p, dtype="float32")[0] for p in rows["path"]]
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt",
                                             return_attention_mask=True)
        with torch.no_grad():
            ids = model.generate(inputs.input_features.to(device, dtype),
                                 attention_mask=inputs.attention_mask.to(device), **gen)
        preds += processor.batch_decode(ids, skip_special_tokens=True)
        print(f"  {min(i + args.batch, len(df))}/{len(df)}", end="\r")
    elapsed = time.time() - t0

    df["raw_prediction"] = preds
    df["ref_norm"] = df["text"].map(normalize_bn)
    df["pred_norm"] = df["raw_prediction"].map(normalize_bn)
    df["wer"] = [jiwer.wer(r, p) if r else None for r, p in zip(df.ref_norm, df.pred_norm)]
    df["cer"] = [jiwer.cer(r, p) if r else None for r, p in zip(df.ref_norm, df.pred_norm)]
    df["script"] = df["raw_prediction"].map(script_flags)
    df["repetition"] = df["pred_norm"].map(repetition)
    df["len_ratio"] = df.pred_norm.str.split().str.len() / df.ref_norm.str.split().str.len()

    # Corpus-level: total edits / total reference words (not an average of per-clip rates)
    metrics = {
        "model": args.model, "run": args.run, "clips": len(df), "decoding": gen,
        "WER%": round(100 * jiwer.wer(list(df.ref_norm), list(df.pred_norm)), 2),
        "CER%": round(100 * jiwer.cer(list(df.ref_norm), list(df.pred_norm)), 2),
        "wrong_script_%": round(100 * (df.script == "wrong_script").mean(), 1),
        "empty_%": round(100 * (df.script == "empty").mean(), 1),
        "repetition_loops_%": round(100 * (df.repetition > 0.3).mean(), 1),
        "over_long_output_%": round(100 * (df.len_ratio > 1.5).mean(), 1),   # hallucinated extra words
        "seconds": round(elapsed, 1),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    out = REPO / "results" / "task2" / args.run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "predictions.csv", index=False, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in metrics.items() if k not in ("decoding",)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
