"""Synthetic clinical-vocabulary TEST (data/make_tts_tests.py): 100 sentences x 6 TTS voices.

Scores, per model:
  WER / CER        pooled over all clips, and per tag (MED, EN, SHORT, untagged) and per voice
  term accuracy    % of listed drug / dose / test names (`terms` column) that appear exactly in the
                   output (after the same normalisation). This is the clinical-safety metric of the
                   Task 3C proposal: a wrong drug name matters more than a wrong filler word.
Only clips marked as wrong in the human listening check (listen_check.csv, ok = n) are excluded.
TEST ONLY: never tune anything on these numbers.

Usage: python task2_asr/eval_tts_sentences.py --backend indicconformer --model <dir> --run tts_indicconformer
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
from common import DATA_ROOT, REPO, normalize_bn  # noqa: E402

TTS_DIR = REPO / "data" / "tts"


def load_asr(backend, model, decoding="rnnt"):
    if backend == "indicconformer":
        from other_asr import IndicConformerASR
        return IndicConformerASR(model, decoding=decoding)
    if backend == "indicconformer_nemo":
        from other_asr import IndicConformerNemoASR
        return IndicConformerNemoASR(model, decoding=decoding)
    from asr import BengaliASR
    return BengaliASR(model, max_new_tokens=440)


def term_found(term, hyp, ignore_spaces=False):
    """A term counts if it starts a word in the output; Bengali suffixes may follow (থাইরয়েড -> থাইরয়েডের).
    ignore_spaces: also accept spacing-only variants (ফলোআপে vs ফলো আপে), which a reader understands.
    (v1 required whole words and wrongly rejected correct inflected terms; fixed before any comparison was made.)"""
    t = normalize_bn(term)
    if ignore_spaces:
        return t.replace(" ", "") in hyp.replace(" ", "")
    return re.search(r"(?:^| )" + re.escape(t), hyp) is not None


def term_hits(terms, hyp):
    """(found strict, found ignoring spaces, total)"""
    terms = [t for t in terms.split("|") if t]
    return (sum(term_found(t, hyp) for t in terms), sum(term_found(t, hyp, True) for t in terms), len(terms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="whisper", choices=["whisper", "indicconformer", "indicconformer_nemo"])
    ap.add_argument("--decoding", default="rnnt", choices=["rnnt", "ctc"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    man = pd.read_csv(TTS_DIR / "sentences_manifest.csv", keep_default_na=False)
    check = TTS_DIR / "listen_check.csv"
    if check.exists():
        bad = set(pd.read_csv(check, keep_default_na=False).query("ok == 'n'").file)
        man = man[~man.file.isin(bad)]
    asr = load_asr(args.backend, args.model, args.decoding)
    clips = [sf.read(DATA_ROOT / "tts_sentences" / f, dtype="float32")[0] for f in man.file]
    hyps = asr.transcribe(clips)

    rows = []
    for (_, r), h in zip(man.iterrows(), hyps):
        ref, hyp = normalize_bn(r["text"]), normalize_bn(h)
        found, found_ns, total = term_hits(r["terms"], hyp)
        rows.append({"id": r["id"], "voice": r["voice"], "role": r["role"], "tags": r["tags"], "ref": ref, "hyp": hyp,
                     "cer": jiwer.cer(ref, hyp), "terms_found": found, "terms_found_nospace": found_ns,
                     "terms_total": total})
    df = pd.DataFrame(rows)
    out = REPO / "results" / "task2" / args.run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "clips.csv", index=False, encoding="utf-8")

    def pooled(sub):
        return {"WER%": round(100 * jiwer.wer(list(sub.ref), list(sub.hyp)), 1),
                "CER%": round(100 * jiwer.cer(list(sub.ref), list(sub.hyp)), 1),
                "term_acc%": round(100 * sub.terms_found.sum() / max(1, sub.terms_total.sum()), 1),
                "term_acc_ignoring_spaces%": round(100 * sub.terms_found_nospace.sum() / max(1, sub.terms_total.sum()), 1),
                "clips": len(sub)}
    summary = {"ALL": pooled(df)}
    for t in ["MED", "EN"]:
        summary[t] = pooled(df[df.tags.str.contains(t)])
    summary["SHORT"] = {k: v for k, v in pooled(df[df.tags.str.contains("SHORT")]).items() if not k.startswith("term")}
    summary["untagged"] = pooled(df[df.tags == ""])
    summary["by_voice"] = {v: pooled(g) for v, g in df.groupby("voice")}
    # the terms most often wrong (out of 6 voices): what a clinician would have to correct
    miss = {}
    for (_, r), (_, d) in zip(man.iterrows(), df.iterrows()):
        for t in filter(None, r["terms"].split("|")):
            if not term_found(t, d["hyp"]):
                miss[t] = miss.get(t, 0) + 1
    summary["most_missed_terms"] = dict(sorted(miss.items(), key=lambda x: -x[1])[:15])
    # clinical fine-tune (make_tts_train.py): test terms the training text contained vs never contained
    split = REPO / "data" / "tts_train" / "split_terms.json"
    if split.exists():
        groups = json.loads(split.read_text(encoding="utf-8"))
        for name, terms in [("seen_in_training", groups["seen_in_training"]), ("unseen", groups["unseen"])]:
            hit = tot = 0
            for (_, r), (_, d) in zip(man.iterrows(), df.iterrows()):
                for t in filter(None, r["terms"].split("|")):
                    if t in terms:
                        tot += 1
                        hit += term_found(t, d["hyp"])
            summary[f"term_acc_{name}%"] = round(100 * hit / max(1, tot), 1)
            summary[f"terms_{name}"] = tot
    (out / "summary.json").write_text(json.dumps({"model": args.model, "backend": args.backend, **summary},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
