"""Task 3: speaker-attributed accuracy of the whole pipeline on the 10 simulated conversations.

These clips have exact reference text per speaker (data/diarization/transcripts), so we can
measure cpWER / cpCER (concatenated minimum-permutation WER, the CHiME-6 metric):
  - concatenate each speaker's words in time order, for reference and system
  - try every mapping of system speakers to reference speakers, keep the one with the fewest errors
  - extra / missing speakers count as all-inserted / all-deleted text
Roles are not evaluated here (read news sentences have no doctor or patient).

Usage: python task3_pipeline/evaluate_sim.py --run final        (all improvements)
       python task3_pipeline/evaluate_sim.py --run baseline --baseline
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "task3_pipeline"))
sys.path.insert(0, str(ROOT / "data"))
from common import DATA_ROOT, normalize_bn  # noqa: E402
from evaluate_pipeline import cp_errors, edits  # noqa: E402,F401
from pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--diarizer", default="pyannote31_recluster",
                    help="the reported sim results used 3.1 + re-clustering (the method designed on this data)")
    args = ap.parse_args()
    pipe = Pipeline(diarizer="pyannote31" if args.baseline else args.diarizer,
                    max_new_tokens=128 if args.baseline else 440, vad_gate=not args.baseline)
    out_dir = ROOT / "results" / "task3" / f"sim_{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, tot = [], {"w": [0, 0], "c": [0, 0], "wa": [0, 0], "ca": [0, 0]}
    for wav in sorted((DATA_ROOT / "diarization" / "wav").glob("*.wav")):
        r = pipe.run(wav)
        ref = json.loads((ROOT / "data" / "diarization" / "transcripts" / f"{wav.stem}.json").read_text(encoding="utf-8"))
        ref_by, hyp_by = {}, {}
        for u in sorted(ref, key=lambda u: u["start"]):
            ref_by[u["speaker"]] = (ref_by.get(u["speaker"], "") + " " + normalize_bn(u["text"])).strip()
        for t in r["turns"]:
            hyp_by[t["speaker"]] = (hyp_by.get(t["speaker"], "") + " " + t["text"]).strip()
        row = {"clip": wav.stem, "ref_spk": len(ref_by), "hyp_spk": len(hyp_by)}
        for key, chars in [("w", False), ("c", True)]:
            e, n = cp_errors(dict(ref_by), dict(hyp_by), chars)
            tot[key][0] += e; tot[key][1] += n
            row["cpWER%" if not chars else "cpCER%"] = round(100 * e / n, 1)
        ref_all = " ".join(normalize_bn(u["text"]) for u in sorted(ref, key=lambda u: u["start"]))
        hyp_all = " ".join(t["text"] for t in r["turns"])
        for key, chars in [("wa", False), ("ca", True)]:
            e, n = edits(ref_all, hyp_all, chars)
            tot[key][0] += e; tot[key][1] += n
        rows.append(row)
        print(row)
    overall = {"cpWER%": round(100 * tot["w"][0] / tot["w"][1], 1), "cpCER%": round(100 * tot["c"][0] / tot["c"][1], 1),
               "WER%_speaker_agnostic": round(100 * tot["wa"][0] / tot["wa"][1], 1),
               "CER%_speaker_agnostic": round(100 * tot["ca"][0] / tot["ca"][1], 1)}
    print("OVERALL", overall)
    (out_dir / "eval.json").write_text(json.dumps({"clips": rows, "OVERALL": overall}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
