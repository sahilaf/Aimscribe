"""Task 2: where do the ASR errors come from? (test-set predictions of a run)

Uses jiwer's word alignment (reference vs prediction) for each clip to measure:
  - boundary deletions: is the FIRST or LAST reference word missing, compared with
    how often a middle word is missing (words lost at segment edges)
  - insertions / over-long output: words that were never said (hallucination)
  - most frequent substitution pairs (what gets confused with what)
  - error rate by utterance length (short clips have little context)

Usage: python task2_asr/analyze_asr_errors.py --run ft_small_test
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import REPO  # noqa: E402


def word_ops(ref, hyp):
    """Per reference word: 'ok' / 'sub' / 'del'; plus inserted words and substitution pairs."""
    out = jiwer.process_words(ref, hyp)
    r, h = ref.split(), hyp.split()
    status, subs, ins = ["ok"] * len(r), [], []
    for chunk in out.alignments[0]:
        if chunk.type == "substitute":
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                status[chunk.ref_start_idx + k] = "sub"
                subs.append((r[chunk.ref_start_idx + k], h[chunk.hyp_start_idx + k]))
        elif chunk.type == "delete":
            for k in range(chunk.ref_start_idx, chunk.ref_end_idx):
                status[k] = "del"
        elif chunk.type == "insert":
            ins += h[chunk.hyp_start_idx:chunk.hyp_end_idx]
    return status, subs, ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run_dir = REPO / "results" / "task2" / args.run
    df = pd.read_csv(run_dir / "predictions.csv").fillna({"pred_norm": ""})

    first_del = last_del = mid_del = mid_total = 0
    subs, ins_count, n = Counter(), 0, 0
    for ref, hyp in zip(df.ref_norm, df.pred_norm):
        if not ref:
            continue
        status, s, ins = word_ops(ref, hyp if hyp else "<empty>")
        n += 1
        first_del += status[0] == "del"
        last_del += len(status) > 1 and status[-1] == "del"
        mid = status[1:-1]
        mid_del += sum(x == "del" for x in mid)
        mid_total += len(mid)
        subs.update(s)
        ins_count += len(ins)

    df["n_words"] = df.ref_norm.str.split().str.len()
    df["len_bin"] = pd.cut(df.n_words, [0, 2, 3, 4, 100], labels=["1-2 words", "3 words", "4 words", "5+ words"])
    by_len = {str(b): round(100 * jiwer.wer(list(g.ref_norm), list(g.pred_norm.fillna(""))), 1)
              for b, g in df.groupby("len_bin", observed=True)}
    total_ref_words = int(df.n_words.sum())
    report = {
        "clips": n,
        "first_word_deleted_%": round(100 * first_del / n, 1),
        "last_word_deleted_%": round(100 * last_del / n, 1),
        "middle_word_deleted_%": round(100 * mid_del / max(1, mid_total), 1),
        "inserted_words_per_100_ref_words": round(100 * ins_count / total_ref_words, 1),
        "WER%_by_utterance_length": by_len,
        "top_substitutions(ref -> hyp, count)": [(a, b, c) for (a, b), c in subs.most_common(15)],
        "worst_examples": df.sort_values("wer", ascending=False)[["text", "raw_prediction", "wer"]].head(8).values.tolist(),
    }
    (run_dir / "error_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
