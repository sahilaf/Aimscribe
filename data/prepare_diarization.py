"""Task 1, Phase 1: build a small Bengali multi-speaker evaluation set with exact labels.

There is no free Bengali conversational corpus with speaker labels, so we
SIMULATE conversations from single-speaker OpenSLR 53 clips (speakers
never used for ASR training). We place every utterance ourselves, so speaker
identity and overlap are exact. Speech boundaries come from Silero VAD run on each
CLEAN single-speaker utterance before mixing (see speech_regions for why).

Each simulated conversation contains:
  - 2 or 3 speakers taking turns with natural pauses (0.2-1.0 s)
  - overlapping turns: next speaker starts before the previous one finishes
  - short "backchannel" utterances dropped inside another speaker's turn
  - different loudness per speaker; background noise on the last clips

Outputs (inside the repo, small):
  data/diarization/rttm/<clip>.rttm         reference "who spoke when"
  data/diarization/transcripts/<clip>.json  one entry per utterance + the words spoken (used in Task 3)
  data/diarization/plots/<clip>.png         timeline picture, overlaps in red
  data/diarization/summary.csv
Audio goes to DATA_ROOT/diarization/wav (2 clips are also copied into data/samples).

Usage:  python data/prepare_diarization.py --n_clips 10
"""
import argparse
import json
import random
import re
import shutil
from collections import defaultdict

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from common import DATA_ROOT, RAW, REPO, SR, extract_zip, normalize_bn, read_transcripts

OUT = REPO / "data" / "diarization"
SHORT_MAX = 1.6  # seconds; utterances shorter than this can be used as backchannels
MIN_UTTS = 10    # a speaker needs enough utterances to fill a conversation
MERGE_GAP_MS = 300  # pauses shorter than this inside an utterance still count as speech (annotator convention)


_VAD = None


def speech_regions(audio):
    """Speech intervals (seconds) in a CLEAN single-speaker clip, via Silero VAD.

    v1 of this script used energy trimming (librosa top_db=30) and labelled the
    whole trimmed clip as speech. Validation showed that was wrong: many OpenSLR
    phone recordings have a noise floor only ~20 dB below the voice, so noisy
    lead-ins, tails and pauses were labelled as speech (two independent VADs,
    pyannote and Silero, found only ~40% of one speaker's "speech").
    """
    global _VAD
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    if _VAD is None:
        _VAD = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(audio), _VAD, sampling_rate=SR, return_seconds=True,
                               min_silence_duration_ms=MERGE_GAP_MS, speech_pad_ms=30)
    return [(t["start"], t["end"]) for t in ts]


def load_utterance(path, target_rms_db=-20.0):
    """Load a clip, cut it to the first..last detected speech, keep the speech
    intervals inside it (pauses >= MERGE_GAP_MS are not speech), loudness-normalise."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    regions = speech_regions(audio)
    if not regions:
        return None
    s, e = regions[0][0], regions[-1][1]
    audio = audio[int(s * SR): int(e * SR)]
    regions = [(a - s, b - s) for a, b in regions]
    audio = audio * (10 ** (target_rms_db / 20) / (np.sqrt(np.mean(audio ** 2)) + 1e-9))
    return {"audio": audio, "regions": regions}


def overlap_seconds(segments):
    """Total time where 2+ speakers talk at once (sweep over start/end events)."""
    events = sorted([(s["start"], 1) for s in segments] + [(s["end"], -1) for s in segments])
    active, last_t, total = 0, 0.0, 0.0
    for t, delta in events:
        if active >= 2:
            total += t - last_t
        active += delta
        last_t = t
    return total


def write_rttm(segments, clip_id, path):
    # RTTM: SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    with open(path, "w", encoding="utf-8") as f:
        for s in sorted(segments, key=lambda x: x["start"]):
            f.write(f"SPEAKER {clip_id} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} "
                    f"<NA> <NA> {s['speaker']} <NA> <NA>\n")


def plot_timeline(segments, clip_id, duration, path):
    speakers = sorted({s["speaker"] for s in segments})
    fig, ax = plt.subplots(figsize=(14, 1 + 0.6 * len(speakers)))
    for s in segments:
        y = speakers.index(s["speaker"])
        ax.barh(y, s["end"] - s["start"], left=s["start"], height=0.6, color=f"C{y}")
    # shade overlap regions
    pts = sorted([(s["start"], 1) for s in segments] + [(s["end"], -1) for s in segments])
    active, last_t = 0, 0.0
    for t, d in pts:
        if active >= 2:
            ax.axvspan(last_t, t, color="red", alpha=0.3)
        active += d
        last_t = t
    ax.set_yticks(range(len(speakers)), speakers)
    ax.set_xlim(0, duration)
    ax.set_xlabel("time (s)   red = overlapping speech")
    ax.set_title(clip_id)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def build_conversation(pool, speakers, rng, target_sec, overlap_prob, backchannel_prob):
    """pool: {speaker: [utterance dicts with 'audio', 'regions', 'text']}.

    Returns (audio, segments, utterances):
      segments    speech-only intervals per speaker -> reference RTTM (for DER)
      utterances  one entry per placed utterance with its text -> reference transcript (Task 3)
    """
    gains = {spk: 10 ** (rng.uniform(-4, 0) / 20) for spk in speakers}
    placed = []                                   # (start_sec, utterance dict, speaker)
    speaker_free_at = {spk: 0.0 for spk in speakers}
    t, prev_spk, prev_start = 0.5, None, 0.0
    queues = {spk: list(pool[spk]) for spk in speakers}

    while t < target_sec and any(queues.values()):
        choices = [s for s in speakers if s != prev_spk and queues[s]] or [s for s in speakers if queues[s]]
        spk = rng.choice(choices)
        utt = queues[spk].pop(0)
        start = t
        if prev_spk and rng.random() < overlap_prob:
            start = max(prev_start + 0.5, t - rng.uniform(0.3, 1.0))   # interrupt the previous speaker
        start = max(start, speaker_free_at[spk] + 0.1)                 # nobody overlaps themselves
        dur = len(utt["audio"]) / SR
        placed.append((start, utt, spk))
        speaker_free_at[spk] = start + dur

        # backchannel: a short reply ("হ্যাঁ"-like) from someone else in the middle of this turn
        if rng.random() < backchannel_prob and dur > 2.0:
            others = [s for s in speakers if s != spk and queues[s]]
            shorts = [(s, k) for s in others for k, u in enumerate(queues[s])
                      if len(u["audio"]) / SR <= SHORT_MAX]
            if shorts:
                bspk, k = rng.choice(shorts)
                butt = queues[bspk].pop(k)
                bstart = max(start + rng.uniform(0.5, dur - 1.0), speaker_free_at[bspk] + 0.1)
                placed.append((bstart, butt, bspk))
                speaker_free_at[bspk] = bstart + len(butt["audio"]) / SR

        prev_spk, prev_start = spk, start
        t = start + dur + rng.uniform(0.2, 1.0)

    total = max(s + len(u["audio"]) / SR for s, u, _ in placed) + 0.5
    mix = np.zeros(int(total * SR), dtype=np.float32)
    segments, utterances = [], []
    for start, utt, spk in placed:
        i = int(start * SR)
        mix[i:i + len(utt["audio"])] += utt["audio"] * gains[spk]
        for a, b in utt["regions"]:
            segments.append({"speaker": spk, "start": round(start + a, 3), "end": round(start + b, 3)})
        utterances.append({"speaker": spk, "start": round(start, 3),
                           "end": round(start + len(utt["audio"]) / SR, 3), "text": utt["text"]})
    return mix, segments, utterances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_clips", type=int, default=10)
    ap.add_argument("--target_sec", type=float, default=75)
    ap.add_argument("--overlap_prob", type=float, default=0.2)
    ap.add_argument("--backchannel_prob", type=float, default=0.15)
    ap.add_argument("--noisy_clips", type=int, default=2, help="last N clips get background noise")
    ap.add_argument("--noise_snr", type=float, default=15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # Use any OpenSLR speaker that is NOT in the ASR training set, so Task 3 can
    # evaluate the fine-tuned ASR on these conversations without leakage.
    train_speakers = set(pd.read_csv(REPO / "data" / "asr" / "train.csv")["speaker"])
    audio_index = extract_zip(RAW / "asr_bengali_0.zip", RAW / "extracted")
    by_speaker = defaultdict(list)
    for r in read_transcripts(RAW / "utt_spk_text.tsv"):
        text = normalize_bn(r["text"])
        if r["utt"] in audio_index and r["speaker"] not in train_speakers and text \
                and not re.search(r"[0-9০-৯A-Za-z]", text):
            by_speaker[r["speaker"]].append({"path": audio_index[r["utt"]], "text": text})
    eligible = sorted(s for s, u in by_speaker.items() if len(u) >= MIN_UTTS)
    rng.shuffle(eligible)
    print(f"Speakers (not in ASR train) with >={MIN_UTTS} utterances: {len(eligible)}")

    for sub in ["rttm", "transcripts", "plots"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    wav_dir = DATA_ROOT / "diarization" / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for i in range(args.n_clips):
        n_spk = 2 if i < int(args.n_clips * 0.6) else 3
        if len(eligible) < n_spk:
            print("Not enough speakers left; stopping.")
            break
        speakers = [eligible.pop() for _ in range(n_spk)]
        pool = {}
        for spk in speakers:
            utts = by_speaker[spk][:]
            rng.shuffle(utts)
            loaded = [(load_utterance(u["path"]), u["text"]) for u in utts]
            pool[spk] = [{**x, "text": text} for x, text in loaded if x is not None]

        mix, segments, utterances = build_conversation(pool, speakers, rng, args.target_sec,
                                                       args.overlap_prob, args.backchannel_prob)
        noisy = i >= args.n_clips - args.noisy_clips
        if noisy:
            sig_power = np.mean(mix[mix != 0] ** 2)
            noise = np_rng.normal(0, np.sqrt(sig_power / 10 ** (args.noise_snr / 10)), len(mix))
            mix = mix + noise.astype(np.float32)
        mix = mix / max(1.0, np.abs(mix).max() / 0.95)

        clip_id = f"bn_conv_{i:02d}"
        sf.write(wav_dir / f"{clip_id}.wav", mix, SR, subtype="PCM_16")
        write_rttm(segments, clip_id, OUT / "rttm" / f"{clip_id}.rttm")
        (OUT / "transcripts" / f"{clip_id}.json").write_text(
            json.dumps(sorted(utterances, key=lambda s: s["start"]), ensure_ascii=False, indent=1),
            encoding="utf-8")
        duration = len(mix) / SR
        plot_timeline(segments, clip_id, duration, OUT / "plots" / f"{clip_id}.png")

        ov = overlap_seconds(segments)
        speech = sum(s["end"] - s["start"] for s in segments)
        summary.append({"clip": clip_id, "speakers": n_spk, "duration_s": round(duration, 1),
                        "utterances": len(utterances), "speech_segments": len(segments),
                        "speech_s": round(speech, 1), "overlap_s": round(ov, 1),
                        "overlap_pct_of_speech": round(100 * ov / speech, 1),
                        "short_utts(<1.6s)": sum(u["end"] - u["start"] <= SHORT_MAX for u in utterances),
                        "noise": f"{args.noise_snr:.0f} dB SNR" if noisy else "clean"})

    df = pd.DataFrame(summary)
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.to_string(index=False))

    sample_dir = REPO / "data" / "samples" / "diarization"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for clip in [df["clip"].iloc[0], df["clip"].iloc[-1]]:  # one clean 2-speaker, one noisy 3-speaker
        shutil.copy(wav_dir / f"{clip}.wav", sample_dir)


if __name__ == "__main__":
    main()
