"""Task 3: Bengali doctor–patient recording -> speaker-attributed, role-labelled transcript.

  audio (any format) -> 16 kHz mono
    -> [SD]  pyannote speaker-diarization-community-1 (Task 1 winner on real multi-speaker Bengali, pyannote 4)
    -> turns: consecutive segments of one speaker merged (gap < MERGE_GAP s, max MAX_TURN s)
    -> [ASR] AI4Bharat IndicConformer 600M (RNN-T) + Silero VAD gate (Task 2 improvement);
             Whisper checkpoints also supported (440-token budget)
    -> [roles] Doctor / Patient / Attendant / Other from what each speaker says (roles.py)
    -> transcript:  Doctor [00:05–00:08]: আপনার কী সমস্যা হচ্ছে?

Why diarize first, then transcribe each turn (instead of transcribing the whole file and
assigning words to speakers afterwards): each ASR input then contains one speaker, and
Whisper's 30 s window is never exceeded. The cost is that a wrong diarization cut
becomes an ASR error too, and overlapped speech is transcribed from mixed audio.

Usage:
  python task3_pipeline/pipeline.py --audio data/consultation/audio/consult_B.wav
  python task3_pipeline/pipeline.py --audio my_visit.m4a --num_speakers 3
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
for sub in ["data", "task1_diarization", "task2_asr", "task3_pipeline"]:
    sys.path.insert(0, str(ROOT / sub))
from asr import BengaliASR  # noqa: E402
from common import DATA_ROOT, normalize_bn  # noqa: E402
from recluster import EMB_MODEL, recluster  # noqa: E402
from roles import assign_roles  # noqa: E402

load_dotenv(ROOT / ".env")
SR = 16000
MERGE_GAP = 0.5   # s: same speaker with a shorter pause = same turn
MAX_TURN = 25.0   # s: Whisper sees at most 30 s at once
# Best on our real consultations (line-level CER 7.2% vs 13.0% tugstugi whisper-medium vs 39.2% our whisper-small
# fine-tune; see notes.md Phase 4b). Download once (gated, accept terms on HF):
#   snapshot_download("ai4bharat/indic-conformer-600m-multilingual", local_dir=<DATA_ROOT>/models/indic-conformer-600m)
# Other choices: --asr_model bengaliAI/tugstugi_bengaliai-asr_whisper-medium | ~/aimscribe_data/models/whisper-small-bn-ft
DEFAULT_ASR = str(DATA_ROOT / "models" / "indic-conformer-600m")


def load_asr(model, max_new_tokens=440, num_beams=1, vad_gate=True):
    """IndicConformer (ONNX, RNN-T) or any Whisper checkpoint, same .transcribe(clips) interface."""
    if "indic-conformer" in str(model):
        from other_asr import IndicConformerASR
        return IndicConformerASR(model, vad_gate=vad_gate)
    return BengaliASR(model, max_new_tokens=max_new_tokens, num_beams=num_beams, vad_gate=vad_gate)


def load_audio(path):
    """Any format (m4a, mp3, wav…) -> mono 16 kHz float32, via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def ts(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


DIARIZERS = {
    # Task 1 winner on real multi-speaker Bengali (DiarBench test DER 19.8 -> 16.7); needs pyannote.audio 4
    "community1": "pyannote/speaker-diarization-community-1",
    "pyannote31": "pyannote/speaker-diarization-3.1",
    # 3.1 + segment re-clustering: the Task 1 improvement on SIMULATED short turns; did not transfer to real
    # meetings (DiarBench dev 18.3 -> 20.8+), kept only to reproduce earlier results
    "pyannote31_recluster": "pyannote/speaker-diarization-3.1",
}


def _from_pretrained(cls, name, token):
    try:
        return cls.from_pretrained(name, token=token)             # pyannote.audio 4
    except TypeError:
        return cls.from_pretrained(name, use_auth_token=token)    # pyannote.audio 3.x


class Pipeline:
    def __init__(self, asr_model=DEFAULT_ASR, threshold=0.70, num_speakers=None, num_beams=1,
                 diarizer="community1", max_new_tokens=440, vad_gate=True):
        """diarizer / max_new_tokens / vad_gate switch the Task 1 and Task 2 improvements
        off, so the whole system can be compared before vs after."""
        from pyannote.audio import Pipeline as PyannotePipeline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = os.environ.get("HF_TOKEN")
        self.sd = _from_pretrained(PyannotePipeline, DIARIZERS[diarizer], token).to(device)
        self.use_recluster = diarizer == "pyannote31_recluster"
        if self.use_recluster:
            from pyannote.audio import Inference, Model
            self.emb = Inference(_from_pretrained(Model, EMB_MODEL, token), window="whole", device=device)
        self.asr = load_asr(asr_model, max_new_tokens, num_beams, vad_gate)
        self.threshold, self.num_speakers, self.diarizer = threshold, num_speakers, diarizer

    def diarize(self, audio, uri):
        inp = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR, "uri": uri}
        kw = {"num_speakers": self.num_speakers} if self.num_speakers else {}
        hyp = self.sd(inp, **kw)
        hyp = getattr(hyp, "speaker_diarization", hyp)   # pyannote 4 wraps the annotation in an output object
        if not self.use_recluster:
            return hyp
        return recluster(hyp, audio, SR, self.emb, self.threshold, self.num_speakers, min_dur=1.0)

    @staticmethod
    def make_turns(ann):
        turns = []
        for seg, _, spk in sorted(ann.itertracks(yield_label=True), key=lambda x: x[0].start):
            prev = turns[-1] if turns else None
            # extend the previous turn only if it's the same speaker (nobody else started in between)
            if prev and prev["speaker"] == spk and seg.start - prev["end"] < MERGE_GAP \
                    and seg.end - prev["start"] <= MAX_TURN:
                prev["end"] = max(prev["end"], seg.end)
                continue
            turns.append({"speaker": spk, "start": seg.start, "end": seg.end})
        for t in turns:   # overlap flag: another speaker active during this turn
            t["overlap"] = any(o is not t and o["speaker"] != t["speaker"] and o["start"] < t["end"] and o["end"] > t["start"]
                               for o in turns)
        return turns

    def run(self, audio_path):
        t0 = time.time()
        audio = load_audio(audio_path)
        uri = Path(audio_path).stem
        ann = self.diarize(audio, uri)
        t_sd = time.time() - t0
        turns = self.make_turns(ann)
        texts = self.asr.transcribe([audio[int(t["start"] * SR):int(t["end"] * SR)] for t in turns])
        for t, x in zip(turns, texts):
            t["text"] = normalize_bn(x)
        turns = [t for t in turns if t["text"]]           # VAD gate / empty output: drop
        t_asr = time.time() - t0 - t_sd

        spk_texts, spk_speech = {}, {}
        for t in turns:
            spk_texts.setdefault(t["speaker"], []).append(t["text"])
            spk_speech[t["speaker"]] = spk_speech.get(t["speaker"], 0) + t["end"] - t["start"]
        roles, feats = assign_roles(spk_texts, spk_speech)
        # two speakers may share a role name (e.g. two attendants): number them
        counts = {}
        names = {}
        for s in sorted(roles, key=lambda s: min(t["start"] for t in turns if t["speaker"] == s)):
            r = roles[s]
            counts[r] = counts.get(r, 0) + 1
            names[s] = r if list(roles.values()).count(r) == 1 else f"{r} {counts[r]}"
        for t in turns:
            t["role"] = names[t["speaker"]]
        return {"uri": uri, "duration_s": round(len(audio) / SR, 2), "turns": turns, "roles": names,
                "role_features": feats, "timing_s": {"diarization": round(t_sd, 1), "asr": round(t_asr, 1),
                                                     "total": round(time.time() - t0, 1)},
                "diarization": ann}


def format_transcript(result):
    lines = []
    for t in result["turns"]:
        mark = " (overlap)" if t["overlap"] else ""
        lines.append(f"{t['role']} [{ts(t['start'])}–{ts(t['end'])}]{mark}: {t['text']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "task3"))
    ap.add_argument("--asr_model", default=DEFAULT_ASR)
    ap.add_argument("--num_speakers", type=int, default=None, help="if known (e.g. asked at the start of the visit)")
    ap.add_argument("--threshold", type=float, default=0.70, help="re-clustering cosine distance (pyannote31_recluster)")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--diarizer", default="community1", choices=list(DIARIZERS))
    ap.add_argument("--baseline", action="store_true",
                    help="switch off the Task 1/2 improvements (pyannote 3.1, 128 tokens, no VAD gate)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252, which cannot print Bengali
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(args.asr_model, args.threshold, args.num_speakers, args.num_beams,
                    diarizer="pyannote31" if args.baseline else args.diarizer,
                    max_new_tokens=128 if args.baseline else 440, vad_gate=not args.baseline)
    for path in args.audio:
        r = pipe.run(path)
        text = format_transcript(r)
        (out / f"{r['uri']}_transcript.txt").write_text(text + "\n", encoding="utf-8")
        with open(out / f"{r['uri']}.rttm", "w", encoding="utf-8") as f:
            r["diarization"].write_rttm(f)
        (out / f"{r['uri']}.json").write_text(json.dumps({k: v for k, v in r.items() if k != "diarization"},
                                                         ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(f"\n===== {r['uri']} ({r['duration_s']} s, processed in {r['timing_s']['total']} s) | roles {r['roles']}")
        print(text)


if __name__ == "__main__":
    main()
