"""Task 3: evaluate the full pipeline against the hand-checked consultation references.

  DER                  who-spoke-when, speaker names ignored (optimal mapping), collar 0.25
  Role error rate      same as DER but WITHOUT mapping: a second of speech counts as correct only
                       if the predicted ROLE equals the true role (Doctor/Patient/Attendant).
                       (pyannote.metrics IdentificationErrorRate)
  WER / CER (all text) everything concatenated in time order, speakers ignored: pure ASR quality
  role-attributed      text of each role compared with that role's reference text, errors summed
  WER / CER            over roles. Text given to the wrong role counts as deleted from the right
                       role and inserted into the wrong one, so it captures SD + roles + ASR together.
  cpWER / cpCER        speaker-aware (CHiME-6): each true speaker's text vs one predicted speaker's text,
                       best one-to-one mapping. Unlike role-attributed WER, this also penalises mixing up
                       two speakers with the same role (Attendant 1 vs Attendant 2).
  Role assignment on the REFERENCE text is also checked, to separate rule failures from ASR noise.

Usage: python task3_pipeline/evaluate_pipeline.py --run final   (reads results/task3/<run>/*.json)
       python task3_pipeline/evaluate_pipeline.py --run tts_final --set tts_consultation
"""
import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import jiwer
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.identification import IdentificationErrorRate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "task3_pipeline"))
from common import normalize_bn  # noqa: E402
from roles import assign_roles  # noqa: E402

def edits(ref, hyp, chars=False):
    """(errors, reference length) at word or character level; handles empty strings."""
    if not ref:
        return (len(hyp) if chars else len(hyp.split())), 0
    if not hyp:
        return (len(ref) if chars else len(ref.split())), (len(ref) if chars else len(ref.split()))
    o = jiwer.process_characters(ref, hyp) if chars else jiwer.process_words(ref, hyp)
    n = o.hits + o.substitutions + o.deletions
    return o.substitutions + o.deletions + o.insertions, n


def base_role(name):
    return name.split(" ")[0]          # "Attendant 2" -> "Attendant"


def cp_errors(ref_by_spk, hyp_by_spk, chars):
    """(errors, ref length) under the best one-to-one speaker mapping; unmatched speakers vs ""."""
    refs, hyps = list(ref_by_spk.values()), list(hyp_by_spk.values())
    n = max(len(refs), len(hyps))
    refs += [""] * (n - len(refs))
    hyps += [""] * (n - len(hyps))
    best = None
    for perm in permutations(range(n)):
        e = N = 0
        for i, j in enumerate(perm):
            ei, ni = edits(refs[i], hyps[j], chars)
            e += ei; N += ni
        if best is None or e < best[0]:
            best = (e, N)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", default="consultation", help="data/<set>/{rttm,transcripts}: consultation | tts_consultation")
    args = ap.parse_args()
    run_dir = ROOT / "results" / "task3" / args.run
    ref_dir = ROOT / "data" / args.set
    report, tot = {}, {k: [0, 0] for k in ["w_all", "c_all", "w_role", "c_role", "w_cp", "c_cp"]}
    der_m, ier_m = DiarizationErrorRate(collar=0.25), IdentificationErrorRate(collar=0.25)

    recs = sorted(p.stem for p in (ref_dir / "transcripts").glob("*.json") if (run_dir / f"{p.stem}.json").exists())
    for rec in recs:
        out = json.loads((run_dir / f"{rec}.json").read_text(encoding="utf-8"))
        refs = sorted(json.loads((ref_dir / "transcripts" / f"{rec}.json").read_text(encoding="utf-8")),
                      key=lambda l: l["start"])
        turns = out["turns"]
        ref_ann = load_rttm(ref_dir / "rttm" / f"{rec}.rttm")[rec]
        # RTTM names speakers ("Attendant 1", "Nurse"); the role error compares ROLES
        spk_role = {l["speaker"]: l["role"] for l in refs}
        ref_role = ref_ann.rename_labels(mapping=spk_role)
        hyp_spk = load_rttm(run_dir / f"{rec}.rttm")[rec]
        hyp_role = Annotation(uri=rec)
        for i, t in enumerate(turns):
            hyp_role[Segment(t["start"], t["end"]), i] = base_role(t["role"])
        der = der_m(ref_ann, hyp_spk)
        ier = ier_m(ref_role, hyp_role)

        ref_all = normalize_bn(" ".join(l["text"] for l in refs))
        hyp_all = " ".join(t["text"] for t in turns)
        r = {"DER%": round(100 * der, 2), "role_error_rate%": round(100 * ier, 2)}
        for key, chars in [("w_all", False), ("c_all", True)]:
            e, n = edits(ref_all, hyp_all, chars)
            tot[key][0] += e; tot[key][1] += n
            r[("WER%" if not chars else "CER%") + "_all_text"] = round(100 * e / n, 1)
        roles = {l["role"] for l in refs} | {base_role(t["role"]) for t in turns}
        for key, chars in [("w_role", False), ("c_role", True)]:
            E = N = 0
            for role in roles:
                rr = normalize_bn(" ".join(l["text"] for l in refs if l["role"] == role))
                hh = " ".join(t["text"] for t in turns if base_role(t["role"]) == role)
                e, n = edits(rr, hh, chars)
                E += e; N += n
            tot[key][0] += E; tot[key][1] += N
            r[("WER%" if not chars else "CER%") + "_role_attributed"] = round(100 * E / N, 1)
        ref_spk = {s: normalize_bn(" ".join(l["text"] for l in refs if l["speaker"] == s)) for s in spk_role}
        hyp_by = {}
        for t in turns:
            hyp_by.setdefault(t["speaker"], []).append(t["text"])
        for key, chars in [("w_cp", False), ("c_cp", True)]:
            e, n = cp_errors(ref_spk, {s: " ".join(x) for s, x in hyp_by.items()}, chars)
            tot[key][0] += e; tot[key][1] += n
            r["cpCER%" if chars else "cpWER%"] = round(100 * e / n, 1)

        # role rules on the REFERENCE text (perfect SD + perfect ASR): are the rules themselves right?
        texts = {}
        speech = {}
        for l in refs:
            texts.setdefault(l["speaker"], []).append(normalize_bn(l["text"]))
            speech[l["speaker"]] = speech.get(l["speaker"], 0) + l["end"] - l["start"]
        oracle_roles, _ = assign_roles(texts, speech)
        r["roles_from_reference_text_correct"] = all(oracle_roles[s] == spk_role[s] for s in oracle_roles)
        r["true_roles"] = spk_role
        r["predicted_roles"] = out["roles"]
        r["processing_s"] = out["timing_s"]["total"]
        r["audio_s"] = out["duration_s"]
        report[rec] = r

    report["OVERALL"] = {
        "DER%": round(100 * abs(der_m), 2), "role_error_rate%": round(100 * abs(ier_m), 2),
        "WER%_all_text": round(100 * tot["w_all"][0] / tot["w_all"][1], 1),
        "CER%_all_text": round(100 * tot["c_all"][0] / tot["c_all"][1], 1),
        "WER%_role_attributed": round(100 * tot["w_role"][0] / tot["w_role"][1], 1),
        "CER%_role_attributed": round(100 * tot["c_role"][0] / tot["c_role"][1], 1),
        "cpWER%": round(100 * tot["w_cp"][0] / tot["w_cp"][1], 1),
        "cpCER%": round(100 * tot["c_cp"][0] / tot["c_cp"][1], 1),
    }
    (run_dir / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
