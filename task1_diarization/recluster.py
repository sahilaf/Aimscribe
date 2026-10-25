"""Task 1 improvement: segment-level re-clustering of pyannote's output.

Diagnosis (see notes.md): baseline DER is ~87% speaker confusion, yet embeddings
computed on WHOLE reference utterances separate speakers perfectly (1-NN 100% on
every clip). pyannote 3.1 clusters embeddings of fixed 10 s windows; with 1-3 s
turns each window holds 2-3 speakers, and assigning the window's local speakers
to global clusters goes wrong.

Fix: keep pyannote's segmentation (speech regions + change points are good:
~3% missed speech), but re-assign speaker labels:
  1. one embedding per system segment, over the whole segment (overlapped parts excluded)
  2. agglomerative clustering (average linkage, cosine) on the reliable segments
     (>= --min_dur seconds), cut by a distance threshold or a known speaker count
  3. short segments go to the nearest cluster centroid instead of forming clusters
  4. two overlapping segments can't share a speaker: the less similar one moves to
     its next-best cluster

Usage:
  python task1_diarization/recluster.py --src baseline --run recluster_t0.70 --threshold 0.70
  python task1_diarization/recluster.py --src baseline --run recluster_oracle --num_speakers oracle
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Inference, Model
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import DATA_ROOT, REPO  # noqa: E402

load_dotenv(REPO / ".env")
EMB_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"  # the same embedding model pyannote 3.1 uses
MIN_EMB = 0.4  # seconds; shorter crops are padded to this length around their centre


def embed(inference, audio, sr, timeline):
    """Embedding of the audio covered by `timeline` (a set of segments, concatenated)."""
    parts = [audio[int(s.start * sr):int(s.end * sr)] for s in timeline]
    x = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    n = int(MIN_EMB * sr)
    if len(x) < n:
        mid = (timeline.extent().start + timeline.extent().end) / 2
        a = min(max(0, int((mid - MIN_EMB / 2) * sr)), max(0, len(audio) - n))   # stay inside the audio
        x = audio[a:a + n]
        if len(x) < n:                                       # clip shorter than MIN_EMB: zero-pad
            x = np.pad(x, (0, n - len(x)))
    e = inference({"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": sr})
    return e / (np.linalg.norm(e) + 1e-9)


def recluster(hyp, audio, sr, inference, threshold, num_speakers, min_dur):
    segs = [(s, t) for s, t, _ in hyp.itertracks(yield_label=True)]
    overlap = hyp.get_overlap()
    embs, durs = [], []
    for seg, _ in segs:
        clean = Timeline([seg]).extrude(overlap)            # drop parts where another speaker talks
        use = clean if clean.duration() >= MIN_EMB else Timeline([seg])
        embs.append(embed(inference, audio, sr, use))
        durs.append(use.duration())
    E, durs = np.stack(embs), np.array(durs)

    reliable = np.where(durs >= min_dur)[0]
    if len(reliable) < 2:                                    # nothing to cluster: keep the input
        return hyp
    Z = linkage(E[reliable], method="average", metric="cosine")
    if num_speakers:
        lab = fcluster(Z, t=num_speakers, criterion="maxclust")
    else:
        lab = fcluster(Z, t=threshold, criterion="distance")
    clusters = sorted(set(lab))
    centroids = np.stack([E[reliable][lab == c].mean(0) for c in clusters])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    sims = E @ centroids.T                                   # every segment vs every centroid

    assign = {}
    for k, i in enumerate(reliable):
        assign[i] = clusters.index(lab[k])
    for i in range(len(segs)):
        if i not in assign:
            assign[i] = int(sims[i].argmax())

    # overlapping segments must belong to different speakers
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if assign[i] == assign[j] and segs[i][0].intersects(segs[j][0]):
                k = i if sims[i, assign[i]] < sims[j, assign[j]] else j
                order = np.argsort(-sims[k])
                assign[k] = int(order[1]) if len(order) > 1 else assign[k]

    out = Annotation(uri=hyp.uri)
    for i, (seg, track) in enumerate(segs):
        out[seg, track] = f"SPK_{assign[i]:02d}"
    return out.support(collar=0.0)                           # merge touching same-speaker segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="baseline", help="run whose segmentation is re-labelled")
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=float, default=0.70, help="cosine distance cut")
    ap.add_argument("--num_speakers", default=None, help="int, or 'oracle'")
    ap.add_argument("--min_dur", type=float, default=1.0, help="segments shorter than this don't form clusters")
    ap.add_argument("--wav_dir", default=str(DATA_ROOT / "diarization" / "wav"))
    ap.add_argument("--ref_dir", default=str(REPO / "data" / "diarization" / "rttm"))
    args = ap.parse_args()

    model = Model.from_pretrained(EMB_MODEL, use_auth_token=os.environ.get("HF_TOKEN"))
    inference = Inference(model, window="whole",
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    src = REPO / "results" / "task1" / args.src / "rttm"
    out_dir = REPO / "results" / "task1" / args.run
    (out_dir / "rttm").mkdir(parents=True, exist_ok=True)

    for rttm in sorted(src.glob("*.rttm")):
        uri = rttm.stem
        hyp = load_rttm(rttm)[uri]
        audio, sr = sf.read(Path(args.wav_dir) / f"{uri}.wav", dtype="float32")
        n = None
        if args.num_speakers == "oracle":
            n = len(load_rttm(Path(args.ref_dir) / f"{uri}.rttm")[uri].labels())
        elif args.num_speakers:
            n = int(args.num_speakers)
        new = recluster(hyp, audio, sr, inference, args.threshold, n, args.min_dur)
        with open(out_dir / "rttm" / f"{uri}.rttm", "w", encoding="utf-8") as f:
            new.write_rttm(f)
        print(f"{uri}: {len(hyp.labels())} -> {len(new.labels())} speakers")
    (out_dir / "config.json").write_text(json.dumps({"method": "segment-level re-clustering",
                                                     "embedding": EMB_MODEL, **vars(args)}, indent=2))


if __name__ == "__main__":
    main()
