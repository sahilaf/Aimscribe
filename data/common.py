"""Shared paths and helpers for dataset preparation.

Large audio lives OUTSIDE the project folder (DATA_ROOT) so it doesn't get
synced to Dropbox or zipped into the submission. Only CSVs, RTTMs and a few
sample files are kept inside the repo.
"""
import os
import re
import unicodedata
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("AIMSCRIBE_DATA", Path.home() / "aimscribe_data"))
RAW = DATA_ROOT / "raw"
SR = 16000  # every model we use (pyannote, Whisper) expects 16 kHz mono

# Characters that are punctuation for Bengali text (danda, quotes, etc.)
_PUNCT = re.compile(r"[।॥,.;:!?\"'“”‘’()\[\]{}<>\-–—…/\\|*~`@#$%^&+=_]")
_SPACES = re.compile(r"\s+")


def normalize_bn(text: str) -> str:
    """Normalize Bengali text so WER/CER only measure real recognition errors.

    - NFC: the same Bengali character can be stored as different Unicode
      sequences (e.g. য় as one code point or as য + nukta). NFC makes them identical.
    - Remove punctuation: Whisper may or may not output '।', and we don't
      want punctuation differences counted as word errors.
    - Remove zero-width joiners and collapse whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("‌", "").replace("‍", "")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def read_transcripts(tsv_path: Path) -> list[dict]:
    """OpenSLR 53 utt_spk_text.tsv -> [{utt, speaker, text}]"""
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                rows.append({"utt": parts[0], "speaker": parts[1], "text": parts[2]})
    return rows


def extract_zip(zip_path: Path, out_dir: Path) -> dict[str, Path]:
    """Extract the OpenSLR zip once and return {utterance_id: flac_path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if not marker.exists():
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        marker.touch()
    return {p.stem: p for p in out_dir.rglob("*.flac")}
