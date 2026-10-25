"""Synthetic TEST sets made with an open Bengali TTS (AI4Bharat Indic Parler-TTS, Apache 2.0, 6 Bengali voices).

TEST ONLY: nothing generated here is ever used for training or tuning. The texts were written and committed
before any system output on them was seen, and the systems are frozen before the first run.

  1. Clinical-vocabulary sentence test (data/tts/tts_test_sentences.tsv)
     100 sentences x 6 voices. References are exact (we wrote the text), and every drug / dose / test
     name is listed in the `terms` column, so medical-term accuracy can be scored, not just CER.
  2. Synthetic consultations (data/tts/consult_scripts.tsv + consult_speakers.tsv)
     6 doctor-patient conversations covering the role cases the real recordings don't:
     patient first, silent patient, two attendants, a child patient, a nurse (Other), many overlaps,
     background noise. Lines are placed on a timeline like prepare_diarization.py, so RTTM is exact.
     timing column: ""       normal turn, 0.25-0.8 s pause after the previous line
                    cut:X    starts X s before the previous line ends (interruption)
                    inside   short reply placed inside the previous line (backchannel)

Caveats (to report): TTS speech is cleaner and more regular than real speech, so scores are optimistic;
the TTS can also mispronounce words, so a sample is checked by ear (listen_check.csv) and clips whose
length is implausible for their text are regenerated with another seed.

Outputs (DATA_ROOT = audio, repo = labels):
  DATA_ROOT/tts_sentences/<id>_<voice>.wav  + data/tts/sentences_manifest.csv
  DATA_ROOT/tts_consultation/<consult>.wav  + data/tts_consultation/{rttm,transcripts,plots}/
Usage (Colab, own venv, see notebooks): python data/make_tts_tests.py --part all
"""
import argparse
import json
import random
import zlib

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from common import DATA_ROOT, REPO, SR
from prepare_diarization import overlap_seconds, plot_timeline, speech_regions, write_rttm

TTS_ID = "ai4bharat/indic-parler-tts"
VOICES = ["Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya"]   # the model card's Bengali speakers
TTS_DIR = REPO / "data" / "tts"
CONSULT_OUT = REPO / "data" / "tts_consultation"
CHARS_PER_S = (4.0, 25.0)   # plausible speaking rate; outside it the clip is regenerated


class TTS:
    def __init__(self, device="cuda"):
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_ID).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(TTS_ID)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.device, self.sr = device, self.model.config.sampling_rate

    def say(self, text, voice, style="speaks at a moderate pace", seed=0):
        """-> 16 kHz mono float32, cut to first..last speech (Silero), with its speech regions."""
        import librosa
        from transformers import set_seed
        desc = f"{voice} {style}. The recording is very clear, with no background noise."
        for attempt in range(3):
            set_seed(seed + 1000 * attempt)
            d = self.desc_tok(desc, return_tensors="pt").to(self.device)
            p = self.tok(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                wav = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                          prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            raw = np.atleast_1d(wav.cpu().numpy().squeeze().astype(np.float32))
            if raw.size < self.sr // 10:      # the TTS occasionally returns (almost) nothing: retry with another seed
                continue
            audio = librosa.resample(raw, orig_sr=self.sr, target_sr=SR)
            regions = speech_regions(audio)
            if not regions:
                continue
            s, e = regions[0][0], regions[-1][1]
            rate = len(text.replace(" ", "")) / max(e - s, 1e-3)
            if CHARS_PER_S[0] <= rate <= CHARS_PER_S[1] or attempt == 2:
                audio = audio[int(s * SR): int(e * SR)]
                audio = audio * (10 ** (-20 / 20) / (np.sqrt(np.mean(audio ** 2)) + 1e-9))
                return {"audio": audio, "regions": [(a - s, b - s) for a, b in regions], "attempts": attempt + 1,
                        "chars_per_s": round(rate, 1)}
        return None


def make_sentences(tts, shard=0, n_shards=1):
    """Several processes can share one GPU (--shard i --n_shards n); clips that already exist are kept (resume).
    The manifest is written by --part manifest once all shards are done."""
    df = pd.read_csv(TTS_DIR / "tts_test_sentences.tsv", sep="\t", keep_default_na=False)
    for _, r in df.iterrows():   # every listed term must literally occur in its sentence
        for t in filter(None, r["terms"].split("|")):
            assert t in r["text"], (r["id"], t)
    wav_dir = DATA_ROOT / "tts_sentences"
    wav_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r, v) for _, r in df.iterrows() for v in VOICES]
    for k, (r, voice) in enumerate(jobs):
        name = f"{r['id']}_{voice}.wav"
        if k % n_shards != shard or (wav_dir / name).exists():
            continue
        out = tts.say(r["text"], voice, seed=zlib.crc32(f"{r['id']}_{voice}".encode()) % 10_000)  # stable across runs
        if out is None:
            print("FAILED", r["id"], voice)
            continue
        sf.write(wav_dir / name, out["audio"], SR, subtype="PCM_16")
        with open(wav_dir / "attempts.tsv", "a", encoding="utf-8") as f:
            f.write(f"{name}\t{out['attempts']}\t{out['chars_per_s']}\n")


def write_manifest():
    df = pd.read_csv(TTS_DIR / "tts_test_sentences.tsv", sep="\t", keep_default_na=False)
    wav_dir = DATA_ROOT / "tts_sentences"
    att = {}
    if (wav_dir / "attempts.tsv").exists():
        for line in (wav_dir / "attempts.tsv").read_text(encoding="utf-8").splitlines():
            name, a, cps = line.split("\t")
            att[name] = (int(a), float(cps))
    rows = []
    for _, r in df.iterrows():
        for voice in VOICES:
            name = f"{r['id']}_{voice}.wav"
            if (wav_dir / name).exists():
                a, cps = att.get(name, (None, None))   # None: made before attempts were logged
                rows.append({**r.to_dict(), "voice": voice, "file": name,
                             "dur": round(sf.info(wav_dir / name).duration, 2), "attempts": a, "chars_per_s": cps})
    man = pd.DataFrame(rows)
    man.to_csv(TTS_DIR / "sentences_manifest.csv", index=False, encoding="utf-8")
    # 60 random clips for a human listening check (is the audio what the text says?)
    man.sample(60, random_state=0)[["file", "text"]].assign(ok="", note="").to_csv(
        TTS_DIR / "listen_check.csv", index=False, encoding="utf-8")
    print(f"sentences: {len(man)} clips, {man.dur.sum() / 60:.1f} min, regenerated {int((pd.to_numeric(man.attempts) > 1).sum())}")
    return man


def build_consult(lines, clips, rng, gains):
    """Place lines on a timeline (see module docstring for the timing column)."""
    placed, t, prev = [], 0.5, None
    free = {}
    for l in lines:
        c = clips[l["line"]]
        dur = len(c["audio"]) / SR
        if l["timing"] == "inside" and prev:
            start = prev["start"] + rng.uniform(0.4, max(0.5, prev["dur"] - dur - 0.3))
        elif l["timing"].startswith("cut:") and prev:
            start = max(prev["start"] + 0.5, prev["start"] + prev["dur"] - float(l["timing"][4:]))
        else:
            start = t
        start = max(start, free.get(l["speaker"], 0) + 0.1)    # nobody overlaps themselves
        free[l["speaker"]] = start + dur
        placed.append({**l, "start": start, "dur": dur})
        if l["timing"] != "inside":
            prev = placed[-1]
        t = max(t, start + dur + rng.uniform(0.25, 0.8))
    total = max(p["start"] + p["dur"] for p in placed) + 0.5
    mix = np.zeros(int(total * SR), dtype=np.float32)
    segments, utterances = [], []
    for p in placed:
        i = int(p["start"] * SR)
        a = clips[p["line"]]["audio"] * gains[p["speaker"]]
        mix[i:i + len(a)] += a
        spk = p["speaker"].replace(" ", "_")          # RTTM fields are space-separated: "Attendant 1" -> "Attendant_1"
        for s, e in clips[p["line"]]["regions"]:
            segments.append({"speaker": spk, "start": round(p["start"] + s, 3), "end": round(p["start"] + e, 3)})
        utterances.append({"line": p["line"], "speaker": spk, "role": p["role"], "start": round(p["start"], 3),
                           "end": round(p["start"] + p["dur"], 3), "text": p["text"]})
    return mix, segments, utterances


def fan_noise(n, rng_np):
    """Low-frequency 'fan / AC' noise: brown noise (integrated white), high-passed to remove drift."""
    x = np.cumsum(rng_np.normal(0, 1, n)).astype(np.float32)
    x -= np.convolve(x, np.ones(800) / 800, mode="same")
    return x / (np.std(x) + 1e-9)


def make_consultations(tts, seed=11):
    lines = pd.read_csv(TTS_DIR / "consult_scripts.tsv", sep="\t", keep_default_na=False)
    spk = pd.read_csv(TTS_DIR / "consult_speakers.tsv", sep="\t", keep_default_na=False)
    rng, rng_np = random.Random(seed), np.random.default_rng(seed)
    for sub in ["rttm", "transcripts", "plots"]:
        (CONSULT_OUT / sub).mkdir(parents=True, exist_ok=True)
    wav_dir = DATA_ROOT / "tts_consultation"
    wav_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for cid, cl in lines.groupby("consult", sort=True):
        cs = spk[spk.consult == cid].set_index("speaker")
        rows = [{**r.to_dict(), "role": cs.loc[r["speaker"], "role"]} for _, r in cl.iterrows()]
        clips = {}
        for k, r in enumerate(rows):
            v = cs.loc[r["speaker"]]
            clips[r["line"]] = tts.say(r["text"], v["voice"], v["style"], seed=seed * 100 + k)
            assert clips[r["line"]] is not None, r["line"]
        gains = {s: 10 ** (rng.uniform(-4, 0) / 20) for s in cs.index}
        mix, segments, utterances = build_consult(rows, clips, rng, gains)
        snr = cs["noise_snr_db"].iloc[0]
        if snr:
            power = np.mean(mix[mix != 0] ** 2)
            mix = mix + fan_noise(len(mix), rng_np) * np.sqrt(power / 10 ** (float(snr) / 10))
        mix = (mix / max(1.0, np.abs(mix).max() / 0.95)).astype(np.float32)
        sf.write(wav_dir / f"{cid}.wav", mix, SR, subtype="PCM_16")
        write_rttm(segments, cid, CONSULT_OUT / "rttm" / f"{cid}.rttm")
        (CONSULT_OUT / "transcripts" / f"{cid}.json").write_text(
            json.dumps(sorted(utterances, key=lambda u: u["start"]), ensure_ascii=False, indent=1), encoding="utf-8")
        plot_timeline(segments, cid, len(mix) / SR, CONSULT_OUT / "plots" / f"{cid}.png")
        speech = sum(s["end"] - s["start"] for s in segments)
        summary.append({"consult": cid, "scenario": cs["scenario"].iloc[0], "speakers": len(cs),
                        "roles": ",".join(sorted(set(cs["role"]))), "lines": len(rows),
                        "duration_s": round(len(mix) / SR, 1), "overlap_s": round(overlap_seconds(segments), 1),
                        "overlap_pct": round(100 * overlap_seconds(segments) / speech, 1),
                        "noise": f"{snr} dB fan" if snr else "clean"})
    df = pd.DataFrame(summary)
    df.to_csv(CONSULT_OUT / "summary.csv", index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["sentences", "consultations", "manifest", "all"], default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()
    if args.part == "manifest":
        write_manifest()
        return
    tts = TTS()
    if args.part in ("sentences", "all"):
        make_sentences(tts, args.shard, args.n_shards)
        if args.n_shards == 1:
            write_manifest()
    if args.part in ("consultations", "all"):
        make_consultations(tts)


if __name__ == "__main__":
    main()
