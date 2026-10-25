"""Task 1: before/after table across runs, on a dev/test split of the 10 clips.

Settings are TUNED on the dev clips (even numbers) and REPORTED on the test clips
(odd numbers); choosing a threshold on the same clips that are reported would overstate
the gain. Both splits contain 2- and 3-speaker and clean/noisy clips.

Usage: python task1_diarization/compare.py baseline oracle_spk thr_0.60 ...
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SPLITS = {"dev": [0, 2, 4, 6, 8], "test": [1, 3, 5, 7, 9], "all": list(range(10))}


def pooled(df, idx):
    sub = df[df["clip"].isin([f"bn_conv_{i:02d}" for i in idx])]
    s = sub[["miss_s", "fa_s", "conf_s", "speech_s"]].sum()
    der = 100 * (s.miss_s + s.fa_s + s.conf_s) / s.speech_s
    spk_ok = (sub["ref_spk"] == sub["hyp_spk"]).sum()
    return round(der, 2), round(100 * s.conf_s / s.speech_s, 2), f"{spk_ok}/{len(sub)}"


def main(runs, collar="0.25"):
    rows = []
    for run in runs:
        df = pd.read_csv(REPO / "results" / "task1" / run / f"der_collar{collar}.csv")
        df = df[df["clip"] != "OVERALL"]
        row = {"run": run}
        for name, idx in SPLITS.items():
            der, conf, spk = pooled(df, idx)
            row[f"{name}_DER%"], row[f"{name}_conf%"], row[f"{name}_spk_ok"] = der, conf, spk
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results" / "task1" / f"comparison_collar{collar}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
