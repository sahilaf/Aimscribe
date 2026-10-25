"""Task 1: overlap-aware diarization candidates, same input / output layout as diarize.py.

pyannote 3.1 misses the second voice in overlapped speech (dev: missed speech 9%, DER inside overlap 38.6% vs 13.4%
elsewhere, and no clustering setting changes it), so the fix has to come from the segmentation side:

  pixit        pyannote/speech-separation-ami-1.0 ("PixIT", Kalda et al. 2024): one model does diarization AND source
               separation, trained on real AMI meetings, so overlapped speech is modelled explicitly.
               Runs in the project environment (pyannote.audio 3.3). Optionally saves one separated track per speaker
               (--save_sources), which Task 3 can transcribe instead of the mixture.
  community1   pyannote/speaker-diarization-community-1: pyannote's 2025 pipeline (better segmentation + VBx-style
               clustering). Needs pyannote.audio 4 -> its own environment.

Usage:
  python task1_diarization/diarize_overlap_aware.py --system pixit --run db_dev_pixit --wav_dir <dev wavs>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.core import Annotation, Segment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import REPO  # noqa: E402

MODELS = {"pixit": "pyannote/speech-separation-ami-1.0", "community1": "pyannote/speaker-diarization-community-1"}


def load(system):
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN")
    try:
        pipe = Pipeline.from_pretrained(MODELS[system], use_auth_token=token)   # pyannote 3.x
    except TypeError:
        pipe = Pipeline.from_pretrained(MODELS[system], token=token)            # pyannote 4 renamed the argument
    return pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(MODELS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--save_sources", default=None, help="folder for PixIT's separated tracks (one wav per speaker)")
    ap.add_argument("--chunk_s", type=float, default=0,
                    help="process long audio in windows of this length (PixIT keeps the whole separation in RAM and was "
                         "killed on 10-15 min clips with 12 GB). Speaker labels then restart per window and are written "
                         "as c<k>_<label>; overlap_fusion.py maps them to 3.1 speakers window by window")
    args = ap.parse_args()
    out_dir = REPO / "results" / "task1" / args.run
    (out_dir / "rttm").mkdir(parents=True, exist_ok=True)
    pipe = load(args.system)
    timings = {}
    for wav in sorted(Path(args.wav_dir).glob("*.wav")):
        if (out_dir / "rttm" / f"{wav.stem}.rttm").exists():   # resume
            continue
        audio, sr = sf.read(wav, dtype="float32")
        t0 = time.time()
        step = int(args.chunk_s * sr) if args.chunk_s and len(audio) > args.chunk_s * sr else len(audio)
        diar, sources = Annotation(uri=wav.stem), None
        for k, a in enumerate(range(0, len(audio), step)):
            x = audio[a:a + step]
            out = pipe({"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": sr, "uri": wav.stem})
            if isinstance(out, tuple):                   # PixIT returns (diarization, separated sources)
                out, sources = out
            part = getattr(out, "speaker_diarization", out)   # pyannote 4 wraps the annotation in an output object
            prefix = f"c{k}_" if step < len(audio) else ""
            for seg, track, label in part.itertracks(yield_label=True):
                diar[Segment(seg.start + a / sr, seg.end + a / sr), track] = prefix + label
        timings[wav.stem] = round(time.time() - t0, 2)
        with open(out_dir / "rttm" / f"{wav.stem}.rttm", "w", encoding="utf-8") as f:
            diar.write_rttm(f)
        if args.save_sources and sources is not None and step == len(audio):
            d = Path(args.save_sources) / wav.stem
            d.mkdir(parents=True, exist_ok=True)
            data = np.asarray(sources.data)               # (frames, speakers)
            for k, label in enumerate(diar.labels()):
                if k < data.shape[1]:
                    sf.write(d / f"{label}.wav", data[:, k], 16000)
        print(f"{wav.stem}: {len(diar.labels())} speakers, {timings[wav.stem]}s")
    (out_dir / "config.json").write_text(json.dumps({"model": MODELS[args.system], "args": vars(args), "timings": timings},
                                                    indent=2))


if __name__ == "__main__":
    main()
