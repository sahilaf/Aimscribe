"""Task 2: does the ASR invent text when nobody is speaking? Before/after the VAD gate.

Probe set = audio that contains NO speech:
  - every gap >= 0.6 s between reference lines in the two consultation recordings
    (real room tone: breathing, fan, chair noise)
  - 3 synthetic clips: digital silence, white noise at -30 dBFS and at -40 dBFS
Metric: how many probes produce any text, and how many words are invented in total.
The same VAD-gated model is also run on the real speech lines (eval_consult.py --vad_gate)
to check the gate doesn't throw away real speech.

Usage: python task2_asr/hallucination_probe.py --model <path>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from asr import SR, BengaliASR  # noqa: E402
from common import REPO, normalize_bn  # noqa: E402

CONSULT = REPO / "data" / "consultation"


def probes():
    out = []
    for rec in ["consult_A", "consult_B"]:
        audio, _ = sf.read(CONSULT / "audio" / f"{rec}.wav", dtype="float32")
        lines = sorted(json.loads((CONSULT / "transcripts" / f"{rec}.json").read_text(encoding="utf-8")),
                       key=lambda l: l["start"])
        busy_until = 0.0
        for l in lines:
            if l["start"] - busy_until >= 0.6:
                a, b = busy_until + 0.1, l["start"] - 0.1          # keep away from speech edges
                out.append((f"{rec} gap {a:.1f}-{b:.1f}s", audio[int(a * SR):int(b * SR)]))
            busy_until = max(busy_until, l["end"])
    rng = np.random.default_rng(0)
    out += [("digital silence 3s", np.zeros(3 * SR, np.float32)),
            ("white noise -30 dBFS 3s", rng.normal(0, 0.0316, 3 * SR).astype(np.float32)),
            ("white noise -40 dBFS 3s", rng.normal(0, 0.01, 3 * SR).astype(np.float32))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    items = probes()
    report = {}
    for gate in [False, True]:
        asr = BengaliASR(args.model, vad_gate=gate)
        hyps = [normalize_bn(h) for h in asr.transcribe([c for _, c in items])]
        name = "with_vad_gate" if gate else "no_gate"
        report[name] = {"probes": len(items), "probes_with_text": sum(bool(h) for h in hyps),
                        "invented_words": sum(len(h.split()) for h in hyps),
                        "examples": [(n, h[:60]) for (n, _), h in zip(items, hyps) if h][:6]}
        print(f"{name:14}: {report[name]['probes_with_text']}/{len(items)} non-speech clips produced text, "
              f"{report[name]['invented_words']} invented words")
        del asr
    out = REPO / "results" / "task2" / "hallucination_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
