"""Task 2: ASR error analysis on the real role-play consultations, line by line.

Each script line is cut from the recording using the hand-checked reference timestamps,
transcribed, and scored. Lines carry the tags from recording_script.md, so errors can be
grouped by the categories the assessment asks about:
  SHORT  (জি / না / হুম / আচ্ছা)    EN  (English words in Bengali)
  MED    (medicine / test names)    OVERLAP (another person talking at the same time)

--pad adds context around each cut. With --pad 0 the cut sits exactly on the
reference boundaries, which is how a diarization segment would be cut in the pipeline.

Usage: python task2_asr/eval_consult.py --model <path or hub id> --run ft_consult --pad 0.0
"""
import argparse
import json
import re
import sys
from pathlib import Path

import jiwer
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import REPO, normalize_bn  # noqa: E402

CONSULT = REPO / "data" / "consultation"
TAGS = {"🔴": "OVERLAP", "🟡": "SHORT", "🔵": "EN", "🟢": "MED", "⚪": "PAUSE"}


def script_tags():
    """{line_id: [tags]} from the markdown tables in recording_script.md."""
    tags = {}
    for line in (CONSULT / "recording_script.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*([AB]\d+)\s*\|\s*[DPA]\s*\|.*\|\s*(.*?)\s*\|\s*$", line)
        if m:
            tags[m.group(1)] = [name for emoji, name in TAGS.items() if emoji in m.group(2)]
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pad", type=float, default=0.0, help="seconds of audio added before/after each line")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128,
                    help="Bengali costs ~2 tokens/character in Whisper's tokenizer; 128 truncates long lines")
    ap.add_argument("--prompt", default=None, help="text prompt (e.g. medical vocabulary) given to the decoder")
    ap.add_argument("--vad_gate", action="store_true", help="skip clips with no speech (Silero VAD) before Whisper")
    ap.add_argument("--vad_crop", action="store_true", help="also crop clips to the detected speech (rejected: cuts words)")
    ap.add_argument("--vad_min_speech_ms", type=int, default=100)
    ap.add_argument("--backend", default="whisper", choices=["whisper", "indicconformer", "indicconformer_nemo", "gemma3n"],
                    help="non-Whisper candidates live in other_asr.py")
    ap.add_argument("--decoding", default="rnnt", choices=["rnnt", "ctc"], help="IndicConformer decoder")
    ap.add_argument("--noise_snr", type=float, default=None,
                    help="add white noise to every line at this SNR (dB), fixed seed: background-noise robustness")
    args = ap.parse_args()

    if args.backend == "indicconformer":
        from other_asr import IndicConformerASR
        asr = IndicConformerASR(args.model, decoding=args.decoding)
    elif args.backend == "indicconformer_nemo":
        from other_asr import IndicConformerNemoASR
        asr = IndicConformerNemoASR(args.model, decoding=args.decoding)
    elif args.backend == "gemma3n":
        from other_asr import Gemma3nASR
        asr = Gemma3nASR(args.model)
    else:
        from asr import BengaliASR
        asr = BengaliASR(args.model, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                         vad_gate=args.vad_gate, vad_crop=args.vad_crop,
                         vad_min_speech_ms=args.vad_min_speech_ms, prompt=args.prompt)

    tags = script_tags()
    rows = []
    for rec in ["consult_A", "consult_B"]:
        audio, sr = sf.read(CONSULT / "audio" / f"{rec}.wav", dtype="float32")
        lines = json.loads((CONSULT / "transcripts" / f"{rec}.json").read_text(encoding="utf-8"))
        clips = [audio[max(0, int((l["start"] - args.pad) * sr)): int((l["end"] + args.pad) * sr)] for l in lines]
        if args.noise_snr is not None:
            import numpy as np
            rng = np.random.default_rng(0)
            clips = [(c + rng.normal(0, np.sqrt(np.mean(c ** 2) / 10 ** (args.noise_snr / 10)), len(c))).astype("float32")
                     for c in clips]
        hyps = asr.transcribe(clips)
        for l, h in zip(lines, hyps):
            ref, hyp = normalize_bn(l["text"]), normalize_bn(h)
            if args.prompt and hyp.startswith(normalize_bn(args.prompt)[:20]):
                hyp = hyp[len(normalize_bn(args.prompt)):].strip()   # some decoders echo the prompt
            base_id = re.sub(r"[ab]$", "", l["line"])
            rows.append({"rec": rec, "line": l["line"], "role": l["role"], "dur": round(l["end"] - l["start"], 2),
                         "tags": ",".join(tags.get(base_id, [])), "ref": ref, "hyp": hyp,
                         "wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)})
    df = pd.DataFrame(rows)
    out = REPO / "results" / "task2" / args.run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lines.csv", index=False, encoding="utf-8")

    def pooled(sub):
        return (round(100 * jiwer.wer(list(sub.ref), list(sub.hyp)), 1),
                round(100 * jiwer.cer(list(sub.ref), list(sub.hyp)), 1), len(sub))
    summary = {"ALL": pooled(df)}
    for t in ["SHORT", "EN", "MED", "OVERLAP"]:
        sub = df[df.tags.str.contains(t)]
        if len(sub):
            summary[t] = pooled(sub)
    summary["untagged"] = pooled(df[df.tags == ""])
    (out / "summary.json").write_text(json.dumps({"model": args.model, "pad": args.pad, "max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams,
                                                  "vad_gate": args.vad_gate, "vad_crop": args.vad_crop, "prompt": args.prompt,
                                                  "noise_snr": args.noise_snr,
                                                  "WER%,CER%,lines": summary}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"=== {args.run} (pad {args.pad}s) : WER% / CER% / lines ===")
    for k, v in summary.items():
        print(f"  {k:9} {v[0]:6.1f} {v[1]:6.1f} {v[2]:3}")


if __name__ == "__main__":
    main()
