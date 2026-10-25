"""Task 1: run pretrained pyannote speaker diarization and save RTTM predictions.

The same script runs the baseline and every improvement experiment, changed only by
flags, so before/after comparisons differ in exactly one setting.

Examples
  python task1_diarization/diarize.py --run baseline
  python task1_diarization/diarize.py --run oracle_spk --num_speakers oracle
  python task1_diarization/diarize.py --run thr_0.6 --clustering_threshold 0.6
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import DATA_ROOT, REPO  # noqa: E402

load_dotenv(REPO / ".env")
MODEL = "pyannote/speaker-diarization-3.1"


def ref_speaker_count(rttm_path):
    return len({line.split()[7] for line in open(rttm_path, encoding="utf-8") if line.strip()})


def load_pipeline(args):
    pipeline = Pipeline.from_pretrained(MODEL, use_auth_token=os.environ.get("HF_TOKEN"))
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # pyannote pipelines expose their tunable hyper-parameters; override only what was asked
    params = pipeline.parameters(instantiated=True)
    if args.clustering_threshold is not None:
        params["clustering"]["threshold"] = args.clustering_threshold
    if args.min_duration_off is not None:
        params["segmentation"]["min_duration_off"] = args.min_duration_off
    if args.min_cluster_size is not None:
        params["clustering"]["min_cluster_size"] = args.min_cluster_size
    pipeline.instantiate(params)
    return pipeline, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="name of this experiment -> results/task1/<run>/")
    ap.add_argument("--wav_dir", default=str(DATA_ROOT / "diarization" / "wav"))
    ap.add_argument("--ref_dir", default=str(REPO / "data" / "diarization" / "rttm"))
    ap.add_argument("--num_speakers", default=None, help="int, or 'oracle' = true count from the reference")
    ap.add_argument("--min_speakers", type=int, default=None)
    ap.add_argument("--max_speakers", type=int, default=None)
    ap.add_argument("--clustering_threshold", type=float, default=None)
    ap.add_argument("--min_duration_off", type=float, default=None)
    ap.add_argument("--min_cluster_size", type=int, default=None)
    args = ap.parse_args()

    out_dir = REPO / "results" / "task1" / args.run
    (out_dir / "rttm").mkdir(parents=True, exist_ok=True)
    pipeline, params = load_pipeline(args)

    timings = {}
    wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    for wav in wavs:
        audio, sr = sf.read(wav, dtype="float32")
        # pass audio in memory: avoids torchaudio file-backend differences between OSes
        inputs = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr, "uri": wav.stem}
        kwargs = {}
        if args.num_speakers == "oracle":
            kwargs["num_speakers"] = ref_speaker_count(Path(args.ref_dir) / f"{wav.stem}.rttm")
        elif args.num_speakers:
            kwargs["num_speakers"] = int(args.num_speakers)
        if args.min_speakers:
            kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers:
            kwargs["max_speakers"] = args.max_speakers

        t0 = time.time()
        diarization = pipeline(inputs, **kwargs)
        elapsed = time.time() - t0
        timings[wav.stem] = {"seconds": round(elapsed, 2), "rtf": round(elapsed / (len(audio) / sr), 4)}
        with open(out_dir / "rttm" / f"{wav.stem}.rttm", "w", encoding="utf-8") as f:
            diarization.write_rttm(f)
        print(f"{wav.stem}: {len(diarization.labels())} speakers, {elapsed:.1f}s")

    config = {"model": MODEL, "args": vars(args), "pipeline_params": params,
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "timings": timings}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
