"""Task 1 improvement: overlap-aware fusion of pyannote 3.1 and PixIT.

Dev diagnosis (notes.md, Phase 6): pyannote 3.1 is accurate where one person speaks but misses the second voice in
overlapped speech (missed speech 14% on the first 15 dev clips, DER inside overlap 41%). PixIT
(pyannote/speech-separation-ami-1.0) models overlap explicitly and halves the missed speech, but adds false alarm and
confusion elsewhere, so on its own it is not better overall.

Fusion = keep pyannote 3.1 everywhere, and borrow from PixIT only what 3.1 lacks:
  1. map PixIT speakers to 3.1 speakers: one-to-one assignment (Hungarian) maximising the time both systems
     give to the pair; PixIT speakers left without a partner are ignored
  2. overlap regions = where PixIT has 2+ speakers active (at least --min_overlap s long)
  3. inside those regions, a mapped speaker that PixIT says is talking but 3.1 does not is ADDED to the 3.1 output
Nothing is ever removed, and additions only happen inside PixIT's overlap regions, so single-speaker accuracy of 3.1 is
kept by construction. This is the classic "overlap detection + second-speaker assignment" idea (Bullock et al., 2020),
with PixIT as the overlap detector and speaker source.

Usage:
  python task1_diarization/overlap_fusion.py --base db_dev_baseline --pixit db_dev_pixit --run db_dev_fusion
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import REPO  # noqa: E402


def speaker_map(base, pixit):
    """{pixit label: base label}, one-to-one, maximising co-occurrence time."""
    b, p = base.labels(), pixit.labels()
    if not b or not p:
        return {}
    co = np.zeros((len(p), len(b)))
    for i, pl in enumerate(p):
        tp = pixit.label_timeline(pl)
        for j, bl in enumerate(b):
            co[i, j] = tp.crop(base.label_timeline(bl), mode="intersection").duration()
    rows, cols = linear_sum_assignment(-co)
    return {p[i]: b[j] for i, j in zip(rows, cols) if co[i, j] > 0}


def windows(pixit, chunk_s):
    """PixIT run in windows (diarize_overlap_aware.py --chunk_s) labels speakers c<k>_<label>; speakers are only
    consistent inside one window, so mapping is done per window. -> [(window Segment or None, sub-annotation)]"""
    groups = {}
    for lab in pixit.labels():
        m = re.match(r"c(\d+)_", lab)
        groups.setdefault(int(m.group(1)) if m else None, []).append(lab)
    if list(groups) == [None]:
        return [(None, pixit)]
    return [(Segment(k * chunk_s, (k + 1) * chunk_s), pixit.subset(labs)) for k, labs in groups.items()]


def fuse(base, pixit, min_overlap=0.0, chunk_s=300.0, adjacent_s=None):
    """adjacent_s: only add a piece that touches (within this gap) a segment the base system already gives to the same
    speaker. Real overlap mostly happens at turn changes (the next speaker starts before the previous one stops), so
    this keeps turn-boundary overlap and rejects PixIT's spurious 'overlap' in the middle of someone else's turn."""
    out = base.copy()
    added, mapping = 0.0, {}
    for win, pix in windows(pixit, chunk_s):
        b = base.crop(win) if win else base
        overlap = Timeline([s for s in pix.get_overlap() if s.duration >= min_overlap]).support()
        m = speaker_map(b, pix)
        mapping.update(m)
        for pl, bl in m.items():
            own = base.label_timeline(bl)
            # where PixIT hears this speaker inside an overlap region, but the base system doesn't
            want = pix.label_timeline(pl).crop(overlap, mode="intersection")
            for seg in want.extrude(own):
                if adjacent_s is not None and not any(
                        seg.start - adjacent_s <= o.end and o.start <= seg.end + adjacent_s for o in own):
                    continue
                out[seg, f"fused_{bl}_{seg.start:.2f}"] = bl
                added += seg.duration
    return out.support(collar=0.0), added, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pyannote 3.1 run (results/task1/<base>/rttm)")
    ap.add_argument("--pixit", required=True, help="PixIT run on the same audio")
    ap.add_argument("--run", required=True)
    ap.add_argument("--min_overlap", type=float, default=0.0, help="ignore PixIT overlap regions shorter than this (s)")
    ap.add_argument("--chunk_s", type=float, default=300.0, help="window length PixIT was run with")
    ap.add_argument("--adjacent_s", type=float, default=None, help="only extend existing turns of that speaker (gap, s)")
    args = ap.parse_args()
    src_b, src_p = (REPO / "results" / "task1" / r / "rttm" for r in (args.base, args.pixit))
    out_dir = REPO / "results" / "task1" / args.run
    (out_dir / "rttm").mkdir(parents=True, exist_ok=True)
    stats = {}
    for f in sorted(src_b.glob("*.rttm")):
        uri = f.stem
        base = load_rttm(f).get(uri, Annotation(uri=uri))
        pf = src_p / f"{uri}.rttm"
        pixit = load_rttm(pf).get(uri, Annotation(uri=uri)) if pf.exists() else Annotation(uri=uri)
        fused, added, mapping = fuse(base, pixit, args.min_overlap, args.chunk_s, args.adjacent_s)
        with open(out_dir / "rttm" / f"{uri}.rttm", "w", encoding="utf-8") as fh:
            fused.write_rttm(fh)
        stats[uri] = {"added_s": round(added, 2), "mapped": len(mapping)}
    (out_dir / "config.json").write_text(json.dumps({"method": "overlap-aware fusion (3.1 + PixIT overlap)",
                                                    **vars(args), "per_clip": stats}, indent=2))
    print(f"{len(stats)} clips, {sum(s['added_s'] for s in stats.values()):.1f} s of overlapped speech added")


if __name__ == "__main__":
    main()
