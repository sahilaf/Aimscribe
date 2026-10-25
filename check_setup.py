"""Phase 0 sanity check: GPU, libraries, Hugging Face token, gated model access."""
import os
from dotenv import load_dotenv

load_dotenv()


def check(name, fn):
    try:
        print(f"OK    {name}: {fn()}")
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}"


def libs():
    import pyannote.audio, transformers, jiwer, librosa
    return f"pyannote {pyannote.audio.__version__}, transformers {transformers.__version__}"


def token():
    from huggingface_hub import whoami
    return whoami(token=os.environ["HF_TOKEN"])["name"]


def gated(repo):
    def _f():
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, "config.yaml", token=os.environ["HF_TOKEN"])
        return "access granted"
    return _f


check("GPU", gpu)
check("Libraries", libs)
check("HF token", token)
for repo in ["pyannote/segmentation-3.0", "pyannote/speaker-diarization-3.1"]:
    check(repo, gated(repo))
