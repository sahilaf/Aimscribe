"""Draft time-stamps for the consultation recordings, to be CHECKED BY HAND in Audacity.

Hand-labelling 45 lines from scratch is slow, so we pre-fill:
  1. diarize the recording (2 speakers; segment-level re-clustering from Task 1)
  2. merge consecutive segments of the same cluster into turns
  3. align the turn sequence to the script's speaker sequence (D, P, D, ...) with
     Needleman-Wunsch (global sequence alignment), trying both cluster->role mappings
  4. write an Audacity label track: one label per script line

The human then listens in Audacity and fixes every boundary. Risk: an annotator who
starts from system output can be biased toward it. Mitigation: every label is
listened to, and overlaps/short replies are checked at the waveform level.

Usage: python data/consultation/prefill_labels.py --run consult_rc
"""
import argparse
import re
from pathlib import Path

from pyannote.database.util import load_rttm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROLES = {"D": "Doctor", "P": "Patient", "A": "Attendant"}


def read_script(script_id):
    """Rows like '| A12 | P | text | tags |' -> [(line_id, role, text)]."""
    rows = []
    for line in (HERE / "recording_script.md").read_text(encoding="utf-8").splitlines():
        m = re.match(rf"\|\s*({script_id}\d+)\s*\|\s*([DPA])\s*\|\s*(.+?)\s*\|", line)
        if m:
            text = re.sub(r"\*\(.*?\)\*", "", m.group(3)).strip()   # drop stage directions
            rows.append((m.group(1), m.group(2), text))
    return rows


def turns_from_rttm(ann):
    turns = []
    for seg, _, label in sorted(ann.itertracks(yield_label=True), key=lambda x: x[0].start):
        if turns and turns[-1]["label"] == label:
            turns[-1]["end"] = max(turns[-1]["end"], seg.end)
        else:
            turns.append({"label": label, "start": seg.start, "end": seg.end})
    return turns


def align(turn_roles, line_roles, match=2, mismatch=-2, gap=-1):
    """Needleman-Wunsch; returns list of (turn_index | None, line_index | None) and the score."""
    n, m = len(turn_roles), len(line_roles)
    S = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        S[i][0] = i * gap
    for j in range(1, m + 1):
        S[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if turn_roles[i - 1] == line_roles[j - 1] else mismatch
            S[i][j] = max(S[i - 1][j - 1] + s, S[i - 1][j] + gap, S[i][j - 1] + gap)
    pairs, i, j = [], n, m
    while i or j:
        s = match if i and j and turn_roles[i - 1] == line_roles[j - 1] else mismatch
        if i and j and S[i][j] == S[i - 1][j - 1] + s:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and S[i][j] == S[i - 1][j] + gap:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1], S[n][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="consult_rc")
    args = ap.parse_args()
    out_dir = HERE / "labels"
    out_dir.mkdir(exist_ok=True)

    for rec in ["A", "B"]:
        uri = f"consult_{rec}"
        script = read_script(rec)
        ann = load_rttm(REPO / "results" / "task1" / args.run / "rttm" / f"{uri}.rttm")[uri]
        turns = turns_from_rttm(ann)
        clusters = sorted(ann.labels())
        script_roles = sorted({r for _, r, _ in script})
        best = None
        for mapping in ({clusters[0]: script_roles[0], clusters[1]: script_roles[1]},
                        {clusters[0]: script_roles[1], clusters[1]: script_roles[0]}):
            pairs, score = align([mapping[t["label"]] for t in turns], [r for _, r, _ in script])
            if best is None or score > best[1]:
                best = (pairs, score, mapping)
        pairs, score, mapping = best

        labels, report = [], []
        for ti, li in pairs:
            if li is None:
                t = turns[ti]
                labels.append((t["start"], t["end"], f"?? EXTRA {mapping[t['label']]} (not in script, check)"))
                report.append(("??", mapping[t["label"]], "EXTRA turn: delete, or merge into a neighbouring line", t["start"], t["end"]))
            elif ti is None:
                lid, role, text = script[li]
                labels.append((0.0, 0.0, f"{lid} {role}: {text}  [NOT FOUND, place manually]"))
                report.append((lid, role, text + "  **NOT FOUND**", None, None))
            else:
                lid, role, text = script[li]
                t = turns[ti]
                flag = "" if mapping[t["label"]] == role else "  [ROLE MISMATCH, check]"
                labels.append((t["start"], t["end"], f"{lid} {role}: {text}{flag}"))
                report.append((lid, role, text + flag, t["start"], t["end"]))

        with open(out_dir / f"{uri}_draft.txt", "w", encoding="utf-8") as f:   # Audacity label format
            for s, e, l in sorted(labels):
                f.write(f"{s:.3f}\t{e:.3f}\t{l}\n")
        print(f"\n=== {uri}: {len(turns)} turns vs {len(script)} script lines | mapping {mapping} ===")
        for lid, role, text, s, e in report:
            ts = f"{s:6.2f}-{e:6.2f}" if s is not None else "   ?   -   ?  "
            print(f"{lid:4} {role} {ts}  {text[:60]}")


if __name__ == "__main__":
    main()
