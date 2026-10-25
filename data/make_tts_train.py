"""Synthetic clinical TRAINING data for fine-tuning IndicConformer 120M (Task 2 / Task 3C), kept apart from every test.

Text: sentences generated from templates (drug + dose + frequency / timing / duration, tests, symptoms, patient and
attendant replies, short replies), so many combinations come from a few hundred hand-written pieces.
Speech: AI4Bharat IndicF5 (MIT), which clones a voice from a short reference clip. References are REAL OpenSLR
TRAIN speakers (data/asr/train.csv) - a different TTS model and different voices from the test sets, which use
Indic Parler-TTS's 6 named voices. So the model cannot score well by learning the test's synthetic voices.

Leakage rules (checked by assert, written to data/tts_train/split_terms.json):
  - no training sentence equals a test sentence (tts_test_sentences.tsv, consult_scripts.tsv)
  - DEV terms (real consult A/B and held-out Script C) never appear in training: the dev score that decides whether
    the fine-tuned model enters the pipeline must not be inflated
  - TEST terms are split in two: SEEN (may appear in training) and UNSEEN (never appear). Term accuracy is reported
    for each group, so learning a drug list can be told apart from generalising to new drug names.

Usage: python data/make_tts_train.py --n 1500    (text + manifest)   then   --synth  (audio, Colab GPU)
"""
import argparse
import json
import random
import re

import pandas as pd

from common import DATA_ROOT, REPO, normalize_bn

OUT = REPO / "data" / "tts_train"

# real consult A / B and Script C words: never in training (dev + held-out real test)
DEV_TERMS = ["প্যারাসিটামল", "অ্যামলোডিপিন", "সিবিসি", "ডেঙ্গু", "এনএস ওয়ান", "ওরস্যালাইন", "অ্যান্টিবায়োটিক",
             "মেট্রোনিডাজল", "জিংক", "স্টুল", "আইবুপ্রোফেন", "ওমিপ্রাজল", "আল্ট্রাসনোগ্রাম", "এন্ডোস্কপি",
             "ডিহাইড্রেশন", "ভাইরাল", "ফিভার", "ডায়াবেটিস", "প্রেশার"]
# test terms that only appear in the synthetic consultations (tts_test_sentences.tsv lists its own in `terms`)
CONSULT_TEST_TERMS = ["সালমিটারল", "ফ্লুটিকাসোন", "গ্লিক্লাজাইড", "ডাইক্লোফেনাক", "অস্টিওআর্থ্রাইটিস",
                      "ফিজিওথেরাপি", "নাইট্রোফুরানটয়েন"]
# extra drugs / tests common in Bangladesh that appear in no test or dev material
EXTRA_DRUGS = ["র‍্যানিটিডিন", "ফ্লুক্লক্সাসিলিন", "সেফুরোক্সিম", "সেফ্রাডিন", "কেটোরোলাক", "ট্রামাডল", "লোরাটাডিন",
               "ডেসলোরাটাডিন", "এমিট্রিপটিলিন", "ক্লোনাজেপাম", "প্রোপ্রানোলল", "বিসোপ্রোলল", "অ্যাটেনোলল",
               "টেলমিসার্টান", "ওলমেসার্টান", "ফুরোসেমাইড", "স্পাইরোনোল্যাকটোন", "রোসুভাস্টাটিন", "গ্লিমেপিরাইড",
               "সিটাগ্লিপটিন", "লিনাগ্লিপটিন", "ডক্সিসাইক্লিন", "লেভোফ্লক্সাসিন", "ক্ল্যারিথ্রোমাইসিন", "ইট্রাকোনাজল",
               "টারবিনাফিন", "আলবেন্ডাজল", "আইভারমেকটিন", "প্রেডনিসোলন", "ডেক্সামেথাসোন", "বুডেসোনাইড",
               "অনডানসেট্রন", "লোপেরামাইড", "ল্যাকটুলোজ", "রাবেপ্রাজল", "ফ্যামোটিডিন", "ওমেগা থ্রি", "ভিটামিন বি কমপ্লেক্স"]
EXTRA_TESTS = ["ইউরিয়া", "ইলেক্ট্রোলাইট", "এইচবিএসএজি", "সিআরপি", "ইউরিক অ্যাসিড", "ট্রপোনিন আই", "ভিডাল টেস্ট",
               "ব্লাড কালচার", "ম্যালেরিয়া টেস্ট", "প্রেগন্যান্সি টেস্ট", "ভিটামিন ডি লেভেল", "পিটি আইএনআর"]
TEST_LIKE = {"সিবিসি", "ইএসআর", "ব্লাড সুগার", "এইচবিএওয়ানসি", "এক্সরে", "ইসিজি", "ইউরিন", "কালচার", "সিরাম ক্রিয়েটিনিন",
             "লিপিড প্রোফাইল", "টিএসএইচ", "ইকোকার্ডিওগ্রাম", "সিটি স্ক্যান", "এমআরআই", "এসজিপিটি", "আলট্রাসাউন্ড", "স্পুটাম"}

GENERIC = {"ক্যাপসুল", "সিরাপ", "ইনহেলার", "ক্রিম", "ড্রপ", "ইউনিট", "রিপোর্ট", "টিকা", "ফলোআপ", "লবণ"}
DIAGNOSIS = {"গ্যাস্ট্রাইটিস", "নিউমোনিয়া", "ইউরিন ইনফেকশন", "গ্যাস্ট্রিক", "মাইগ্রেন", "অ্যাজমা", "স্ট্রোক", "অ্যালার্জি",
             "অস্টিওআর্থ্রাইটিস", "থাইরয়েড", "লিভার", "ডায়াবেটিস", "ব্লাড প্রেশার", "প্রেশারের"}
PROCEDURE = {"ফিজিওথেরাপি", "নেবুলাইজার"}
EXTRA_DX = ["টাইফয়েড", "জন্ডিস", "আর্থ্রাইটিস", "সাইনোসাইটিস", "টনসিলাইটিস", "অ্যানিমিয়া", "হেপাটাইটিস", "আলসার",
            "ব্রংকাইটিস", "একজিমা", "হার্নিয়া", "কিডনি স্টোন"]
EXTRA_PROC = ["ড্রেসিং", "ইনজেকশন", "স্যালাইন", "সেলাই", "এক্সারসাইজ"]
DOSES = ["ফাইভ হান্ড্রেড মিলিগ্রাম", "টু হান্ড্রেড মিলিগ্রাম", "টেন মিলিগ্রাম", "টুয়েন্টি মিলিগ্রাম", "ফোর্টি মিলিগ্রাম",
         "পঞ্চাশ মিলিগ্রাম", "পাঁচশো মিলিগ্রাম", "একশো মিলিগ্রাম", "আড়াইশো মিলিগ্রাম", "সেভেন্টি ফাইভ", "ওয়ান ফিফটি",
         "এক চামচ", "দুই চামচ", "আধা চামচ", "দুই চাপ", "একটা ট্যাবলেট", "একটা ক্যাপসুল"]
FREQ = ["দিনে একবার", "দিনে দুইবার", "দিনে তিনবার", "আট ঘণ্টা পরপর", "বারো ঘণ্টা পরপর", "ছয় ঘণ্টা পরপর",
        "রাতে একটা", "সকালে একটা", "সকালে আর রাতে"]
TIMING = ["খাওয়ার পরে", "খাওয়ার আগে", "খালি পেটে", "ঘুমানোর আগে", "নাস্তার পরে", "দুপুরে খাওয়ার পর"]
DURATION = ["তিন দিন", "পাঁচ দিন", "সাত দিন", "দশ দিন", "চৌদ্দ দিন", "এক মাস", "তিন মাস", "ছয় মাস"]
SYMPTOMS = ["জ্বর", "কাশি", "মাথা ব্যথা", "পেট ব্যথা", "বুকে ব্যথা", "গলা ব্যথা", "বমি", "পাতলা পায়খানা", "শ্বাসকষ্ট",
            "চুলকানি", "কোমরে ব্যথা", "হাঁটুতে ব্যথা", "মাথা ঘোরা", "দুর্বলতা", "অরুচি", "ঘুমের সমস্যা"]
SINCE = ["দুই দিন", "তিন দিন", "এক সপ্তাহ", "দশ দিন", "দুই সপ্তাহ", "এক মাস", "কয়েক মাস"]
SHORT = ["জি", "হ্যাঁ", "না", "আচ্ছা", "ঠিক আছে", "হুম", "জি স্যার", "জি ম্যাডাম", "বুঝেছি", "একটু একটু", "না এখনো না",
         "জি খেয়েছি", "আচ্ছা ঠিক আছে", "হ্যাঁ আছে"]

T_DRUG = [("D", "{drug} {dose} {freq} খাবেন"), ("D", "{drug} {dose} {timing} খাবেন"),
          ("D", "{drug} {dose} {freq} {duration} খাবেন"), ("D", "আপনাকে {drug} {dose} দিচ্ছি, {freq}"),
          ("D", "{drug} {freq} {timing}, {duration} চলবে"), ("D", "{drug} আপাতত বন্ধ রাখবেন"),
          ("D", "{drug} এর সাথে {drug2} যোগ করছি"), ("D", "{drug} {dose} {duration} পুরো কোর্স শেষ করবেন"),
          ("P", "আমি {drug} খাই, {freq}"), ("P", "আমি আগে {drug} খেয়েছিলাম"), ("P", "ফার্মেসি থেকে {drug} দিয়েছিল"),
          ("P", "{drug} খেলে কি কোনো সমস্যা হবে"), ("A", "উনি প্রতিদিন {drug} {dose} খান"),
          ("A", "ওকে {drug} সিরাপ খাওয়াচ্ছি"), ("A", "আব্বা {drug} খাওয়া বন্ধ করে দিয়েছেন")]
T_TEST = [("D", "একটা {test} করিয়ে আনবেন"), ("D", "{test} আর {test2} করাবেন"), ("D", "{test} রিপোর্টটা দেখান"),
          ("D", "আজকেই একটা {test} করা দরকার"), ("P", "{test} কি করাতে হবে"), ("P", "{test} রিপোর্ট নিয়ে এসেছি"),
          ("A", "উনার {test} গত মাসে করানো হয়েছিল")]
T_DX = [("D", "মনে হচ্ছে {dx} হয়েছে"), ("D", "এটা {dx} এর লক্ষণ"), ("D", "আপনার {dx} আছে, নিয়মিত ওষুধ খাবেন"),
        ("P", "আমার আগে থেকেই {dx} আছে"), ("A", "উনার অনেক দিনের {dx}"), ("D", "{proc} লাগবে, সপ্তাহে দুইবার"),
        ("D", "আজকে একটা {proc} দিচ্ছি"), ("P", "{proc} কি নিতেই হবে")]
T_SYM = [("P", "আমার {since} ধরে {symptom}"), ("P", "রাতে {symptom} বেশি হয়"), ("A", "ওর {since} ধরে {symptom}"),
         ("A", "আম্মার {symptom} কমছে না"), ("D", "{symptom} কতদিন ধরে হচ্ছে"), ("D", "{symptom} বাড়লে সাথে সাথে আসবেন")]


def test_terms():
    s = pd.read_csv(REPO / "data" / "tts" / "tts_test_sentences.tsv", sep="\t", keep_default_na=False)
    terms = {t for x in s.terms for t in x.split("|") if t}
    # dose / number words are shared vocabulary, not drug names: they are not split, but test sentences stay unseen
    numbers = {t for t in terms if re.fullmatch(r"(ফাইভ|টু|টেন|ওয়ান|সেভেন্টি|টুয়েন্টি|ফিফটি|হান্ড্রেড| |এইটি)+", t)
               or t == "মিলিগ্রাম"}
    return sorted((terms | set(CONSULT_TEST_TERMS)) - numbers), s.text.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--synth", action="store_true", help="generate audio with IndicF5 (needs GPU)")
    ap.add_argument("--only_ids", default=None, help="CSV with an id column: synthesise only these sentences")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    terms, test_sents = test_terms()
    consult = pd.read_csv(REPO / "data" / "tts" / "consult_scripts.tsv", sep="\t", keep_default_na=False).text.tolist()
    blocked_sents = {normalize_bn(x) for x in test_sents + consult}
    # dosage-form / generic words (ক্যাপসুল, সিরাপ, ড্রপ...) are everyday clinic vocabulary used by the templates:
    # always "seen"; only drug, test and diagnosis NAMES are split
    generic = [t for t in terms if t in GENERIC]
    candidates = [t for t in terms if t not in GENERIC and not any(d in t or t in d for d in DEV_TERMS)]
    rng.shuffle(candidates)
    seen = sorted(candidates[: len(candidates) // 2] + generic)
    unseen = sorted(set(terms) - set(seen))                    # includes every term that touches a dev term
    held_out = unseen + DEV_TERMS

    def allowed(x):          # e.g. "ব্লাড কালচার" contains the held-out "কালচার"
        return not any(h in x for h in held_out)
    other = TEST_LIKE | DIAGNOSIS | PROCEDURE | GENERIC | {"কালচার", "লিপিড প্রোফাইল", "ইউরিন"}
    drugs = [t for t in [t for t in seen if t not in other] + EXTRA_DRUGS if allowed(t)]
    tests = [t for t in [t for t in seen if t in TEST_LIKE | {"কালচার", "লিপিড প্রোফাইল"}] + EXTRA_TESTS if allowed(t)]
    dxs = [t for t in [t for t in seen if t in DIAGNOSIS] + EXTRA_DX if allowed(t)]
    procs = [t for t in [t for t in seen if t in PROCEDURE] + EXTRA_PROC if allowed(t)]

    def fill(tpl):
        return tpl.format(drug=rng.choice(drugs), drug2=rng.choice(drugs), dose=rng.choice(DOSES), freq=rng.choice(FREQ),
                          timing=rng.choice(TIMING), duration=rng.choice(DURATION), test=rng.choice(tests),
                          test2=rng.choice(tests), symptom=rng.choice(SYMPTOMS), since=rng.choice(SINCE),
                          dx=rng.choice(dxs), proc=rng.choice(procs))

    pool = [(T_DRUG, 0.5), (T_TEST, 0.17), (T_DX, 0.1), (T_SYM, 0.15), (None, 0.08)]
    rows, seen_text = [], set()
    while len(rows) < args.n:
        group = rng.choices([p[0] for p in pool], weights=[p[1] for p in pool])[0]
        role, text = (rng.choice("PA"), rng.choice(SHORT)) if group is None else rng.choice(group)
        text = text if group is None else fill(text)
        key = normalize_bn(text)
        if key in blocked_sents or (key in seen_text and group is not None):
            continue
        seen_text.add(key)
        rows.append({"id": f"S{len(rows):05d}", "role": role, "text": text})
    df = pd.DataFrame(rows)

    # leakage checks
    norm = df.text.map(normalize_bn)
    assert not set(norm) & blocked_sents, "a training sentence equals a test sentence"
    for t in unseen + DEV_TERMS:
        assert not norm.str.contains(normalize_bn(t), regex=False).any(), f"held-out term in training: {t}"

    # reference voices: one clean 3-8 s clip per OpenSLR TRAIN speaker
    tr = pd.read_csv(REPO / "data" / "asr" / "train.csv")
    refs = (tr[(tr.duration >= 3) & (tr.duration <= 8)].sort_values("duration", ascending=False)
            .groupby("speaker").head(1)[["speaker", "path", "text"]])
    df["ref_speaker"] = [refs.iloc[i % len(refs)].speaker for i in range(len(df))]
    df = df.merge(refs.rename(columns={"path": "ref_path", "text": "ref_text"}), left_on="ref_speaker", right_on="speaker")
    df = df.drop(columns="speaker").sort_values("id")
    df.to_csv(OUT / "train_sentences.tsv", sep="\t", index=False)
    (OUT / "split_terms.json").write_text(json.dumps({"seen_in_training": seen, "unseen": unseen, "dev_terms_excluded": DEV_TERMS},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(df)} sentences, {df.ref_speaker.nunique()} reference voices | test terms: {len(seen)} seen, {len(unseen)} unseen")
    print(df.role.value_counts().to_dict())

    if args.synth:
        if args.only_ids:   # the reported run stopped at 1,058 clips for time: re-create exactly those
            df = df[df.id.isin(set(pd.read_csv(args.only_ids).id))]
        synth(df)


def synth(df):
    """IndicF5: text + reference clip (+ its transcript) -> 24 kHz speech in that voice; saved as 16 kHz mono."""
    import librosa
    import numpy as np
    import soundfile as sf
    from transformers import AutoModel
    import torch
    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to("cuda")                     # from_pretrained loads on CPU: 3 clips took 24 min there
    # IndicF5 calls f5_tts infer_process with its default 32 flow-matching steps (~3 clips/min on a shared T4);
    # 16 steps halves the time at a small quality cost, fine for training data (env F5_NFE, default 16)
    import functools
    import os
    import sys
    mod = sys.modules[type(model).__module__]
    mod.infer_process = functools.partial(mod.infer_process, nfe_step=int(os.environ.get("F5_NFE", 16)))
    wav_dir = DATA_ROOT / "tts_train"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in df.iterrows():
        out = wav_dir / f"{r['id']}.wav"
        if not out.exists():
            audio = model(r["text"], ref_audio_path=str(DATA_ROOT / r["ref_path"]), ref_text=r["ref_text"])
            audio = np.asarray(audio)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            sf.write(out, librosa.resample(audio.astype(np.float32), orig_sr=24000, target_sr=16000), 16000, subtype="PCM_16")
        rows.append({"audio_filepath": str(out), "duration": round(sf.info(out).duration, 2),
                     "text": normalize_bn(r["text"]), "lang": "bn"})
    with open(OUT / "manifest_tts.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"audio: {len(rows)} clips, {sum(x['duration'] for x in rows) / 3600:.2f} h")


if __name__ == "__main__":
    main()
