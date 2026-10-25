"""Task 1: Diarization Error Rate (DER) of a run against the reference RTTMs.

DER = (missed speech + false alarm + speaker confusion) / total reference speech

  missed       reference speech the system labelled as silence
  false alarm  system said "speech" where there was none
  confusion    speech found, but attributed to the wrong speaker

Overlapping speech counts once per speaker, so two people talking for 1 s adds
2 s to the total. A system that outputs one speaker there gets 1 s "missed".

We report two settings:
  collar=0.25  standard: ignore +/-0.25 s around every reference boundary (human
               annotators can't agree on boundaries more precisely than that)
  collar=0.0   strict: every millisecond counts

Usage: python task1_diarization/evaluate.py --run baseline
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import REPO  # noqa: E402


def evaluate(ref_dir, hyp_dir, collar, overlap_dir=None):
    """overlap_dir: RTTMs of the regions where 2+ reference speakers talk (prepare_diarbench.py). If given,
    DER is also scored INSIDE those regions only and OUTSIDE them, so an overlap fix can be measured where it acts."""
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    m_ov, m_rest = DiarizationErrorRate(collar=collar), DiarizationErrorRate(collar=collar)
    rows = []
    for ref_path in sorted(Path(ref_dir).glob("*.rttm")):
        uri = ref_path.stem
        ref = load_rttm(ref_path)[uri]
        hyp_file = Path(hyp_dir) / f"{uri}.rttm"
        # an empty RTTM (system found no speech) is a valid, terrible prediction
        hyp = (load_rttm(hyp_file).get(uri) if hyp_file.exists() else None) or Annotation(uri=uri)
        d = metric(ref, hyp, detailed=True)
        total = d["total"]
        split = {}
        if overlap_dir:
            ov = load_rttm(Path(overlap_dir) / f"{uri}.rttm").get(uri, Annotation(uri=uri)).get_timeline().support()
            whole = Timeline([Segment(0, max(ref.get_timeline().extent().end, hyp.get_timeline().extent().end
                                             if hyp else 0) + 1)], uri=uri)
            rest = whole.extrude(ov)
            split["DER_overlap%"] = 100 * m_ov(ref, hyp, uem=ov) if ov else None
            split["DER_rest%"] = 100 * m_rest(ref, hyp, uem=rest)
        rows.append({**split,
            "clip": uri,
            "ref_spk": len(ref.labels()),
            "hyp_spk": len(hyp.labels()),
            "DER%": 100 * d["diarization error rate"],
            "miss%": 100 * d["missed detection"] / total,
            "FA%": 100 * d["false alarm"] / total,
            "conf%": 100 * d["confusion"] / total,
            # absolute seconds, so subsets of clips can be pooled correctly later
            "miss_s": d["missed detection"], "fa_s": d["false alarm"],
            "conf_s": d["confusion"], "speech_s": total,
        })
    df = pd.DataFrame(rows)
    # Overall numbers are pooled over all clips (sum of errors / sum of speech), not
    # a mean of per-clip percentages, which is how pyannote.metrics reports them.
    totals = metric.accumulated_
    overall = {"clip": "OVERALL", "ref_spk": None, "hyp_spk": None,
               "DER%": 100 * abs(metric),
               "miss%": 100 * totals["missed detection"] / totals["total"],
               "FA%": 100 * totals["false alarm"] / totals["total"],
               "conf%": 100 * totals["confusion"] / totals["total"],
               "miss_s": totals["missed detection"], "fa_s": totals["false alarm"],
               "conf_s": totals["confusion"], "speech_s": totals["total"]}
    if overlap_dir:
        overall["DER_overlap%"] = 100 * abs(m_ov)
        overall["DER_rest%"] = 100 * abs(m_rest)
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True).round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref_dir", default=str(REPO / "data" / "diarization" / "rttm"))
    ap.add_argument("--overlap_dir", default=None, help="reference overlap regions -> also DER inside / outside overlap")
    args = ap.parse_args()
    run_dir = REPO / "results" / "task1" / args.run
    for collar in (0.25, 0.0):
        df = evaluate(args.ref_dir, run_dir / "rttm", collar, args.overlap_dir)
        df.to_csv(run_dir / f"der_collar{collar}.csv", index=False)
        print(f"\n=== {args.run} | collar={collar} ===")
        print(df.drop(columns=["miss_s", "fa_s", "conf_s", "speech_s"]).to_string(index=False))


if __name__ == "__main__":
    main()
