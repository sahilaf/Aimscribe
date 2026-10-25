"""Turn the hand-checked Audacity labels into reference files.

Input : data/consultation/labels/consult_X.txt   (Audacity: Export Labels; lines "start<TAB>end<TAB>A12 P: text")
Output: data/consultation/rttm/consult_X.rttm            speakers named by role (Doctor/Patient/Attendant), for DER
        data/consultation/transcripts/consult_X.json     [{line, speaker, role, start, end, text}], for WER/CER in Task 3

Usage: python data/consultation/labels_to_reference.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROLES = {"D": "Doctor", "P": "Patient", "A": "Attendant"}
LABEL = re.compile(r"^([AB]\d+[a-z]?)\s+([DPA]):\s*(.+?)\s*$")   # A17a/A17b: a script line split in two


def main():
    for sub in ["rttm", "transcripts"]:
        (HERE / sub).mkdir(exist_ok=True)
    for path in sorted((HERE / "labels").glob("consult_?.txt")):     # the checked files, not *_draft.txt
        uri = path.stem
        entries, problems = [], []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            start, end, label = line.split("\t", 2)
            m = LABEL.match(label)
            if not m or "[" in label or label.startswith("??"):
                problems.append(f"line {n}: '{label}' (remove flags like [NOT FOUND] / ?? EXTRA once fixed)")
                continue
            start, end = float(start), float(end)
            if end <= start:
                problems.append(f"line {n}: end <= start")
                continue
            entries.append({"line": m.group(1), "speaker": ROLES[m.group(2)], "role": ROLES[m.group(2)],
                            "start": round(start, 3), "end": round(end, 3), "text": m.group(3)})
        if problems:
            print(f"{uri}: fix these first:\n  " + "\n  ".join(problems))
            continue
        entries.sort(key=lambda e: e["start"])
        with open(HERE / "rttm" / f"{uri}.rttm", "w", encoding="utf-8") as f:
            for e in entries:
                f.write(f"SPEAKER {uri} 1 {e['start']:.3f} {e['end'] - e['start']:.3f} <NA> <NA> {e['speaker']} <NA> <NA>\n")
        (HERE / "transcripts" / f"{uri}.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{uri}: {len(entries)} lines -> rttm + transcript written")


if __name__ == "__main__":
    main()
