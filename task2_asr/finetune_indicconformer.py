"""Task 2 / 3C: fine-tune AI4Bharat IndicConformer 120M (Bengali, hybrid CTC / RNN-T) on clinical speech.

The 600M multilingual model used in the pipeline is an ONNX export and cannot be trained; the 120M Bengali .nemo
checkpoint can (needs AI4Bharat's NeMo fork, branch nemo-v2, own environment).

Training data (NeMo manifests: one JSON per line with audio_filepath, duration, text, lang):
  - synthetic clinical sentences from data/make_tts_train.py (IndicF5, real OpenSLR voices, no test sentences,
    held-out test terms never seen)
  - real general Bengali: our OpenSLR 53 train split (data/asr/train.csv), so ordinary speech is not forgotten
Dev for checkpoint choice: OpenSLR dev speakers (held out from train, as in finetune_whisper.py). The real
consultations A + B decide afterwards whether the model enters the pipeline (rule fixed in notes.md).

Usage (Colab, venv_nemo):
  python task2_asr/finetune_indicconformer.py --inspect            # print the model's data config
  python task2_asr/finetune_indicconformer.py --max_steps 1500 --lr 5e-5 --run icft_120m
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from common import DATA_ROOT, REPO, normalize_bn  # noqa: E402

BASE = DATA_ROOT / "models" / "indicconformer-bn-120m" / "indicconformer_stt_bn_hybrid_rnnt_large.nemo"


def openslr_manifests(dev_speakers=6, seed=7):
    """Our OpenSLR train split -> train / dev manifests, dev = whole held-out speakers."""
    import pandas as pd
    tr = pd.read_csv(REPO / "data" / "asr" / "train.csv")
    spk = sorted(tr.speaker.unique())
    random.Random(seed).shuffle(spk)
    dev = set(spk[:dev_speakers])
    out = {}
    for name, part in [("train", tr[~tr.speaker.isin(dev)]), ("dev", tr[tr.speaker.isin(dev)])]:
        out[name] = [{"audio_filepath": str(DATA_ROOT / r.path), "duration": float(r.duration),
                      "text": normalize_bn(r.text), "lang": "bn"} for r in part.itertuples()]
    return out


def write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--run", default="icft_120m")
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--real_ratio", type=float, default=1.0, help="OpenSLR clips per synthetic clip (forgetting guard)")
    ap.add_argument("--val_every", type=int, default=250)
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    import pytorch_lightning as pl               # NeMo models are pytorch_lightning modules (not lightning.pytorch)
    import torch
    from omegaconf import OmegaConf, open_dict

    model = nemo_asr.models.ASRModel.restore_from(str(BASE))
    if args.inspect:
        print(OmegaConf.to_yaml(model.cfg.train_ds)); print(OmegaConf.to_yaml(model.cfg.optim))
        print("tokenizer:", type(model.tokenizer).__name__, "| decoders:", getattr(model, "cur_decoder", None))
        return

    out = REPO / "results" / "task2" / args.run
    out.mkdir(parents=True, exist_ok=True)
    tts = [json.loads(l) for l in open(REPO / "data" / "tts_train" / "manifest_tts.jsonl", encoding="utf-8")]
    real = openslr_manifests()
    rng = random.Random(0)
    train = tts + rng.sample(real["train"], min(len(real["train"]), int(args.real_ratio * len(tts))))
    rng.shuffle(train)
    write(train, out / "train_manifest.jsonl")
    write(real["dev"], out / "dev_manifest.jsonl")

    with open_dict(model.cfg):
        for split, path, shuffle in [("train_ds", out / "train_manifest.jsonl", True),
                                     ("validation_ds", out / "dev_manifest.jsonl", False)]:
            ds = model.cfg.get(split) or OmegaConf.create({})
            ds.manifest_filepath = str(path)
            ds.batch_size = args.batch
            ds.shuffle = shuffle
            ds.sample_rate = 16000
            ds.is_tarred = False
            ds.tarred_audio_filepaths = None
            ds.is_concat = False                 # AI4Bharat trained on a LIST of per-dataset manifests (concat)
            ds.return_language_id = True         # multi-softmax head: every manifest line carries "lang": "bn"
            ds.max_duration = 20.0
            ds.num_workers = 2
            model.cfg[split] = ds
        model.cfg.optim.lr = args.lr
        model.cfg.optim.sched = {"name": "CosineAnnealing", "warmup_steps": 100, "min_lr": args.lr / 20,
                                 "max_steps": args.max_steps}
    # The RNN-T loss cannot be trained here: NeMo's numba CUDA kernel cannot attach to Colab's CUDA 13 driver (numba
    # 0.58) or is rejected by newer numba, and the pure-PyTorch RNN-T loss did not finish one step in 4 min on the T4.
    # So the CTC branch of the hybrid model is fine-tuned (CTC weight 1, RNN-T loss switched off); the encoder is
    # shared and trained either way, and the fine-tuned model is decoded with CTC (zero-shot CTC: consult CER 11.1).
    import types

    import torch.nn as nn

    class NoRNNTLoss(nn.Module):
        """Zero RNN-T loss with the interface the fused multi-softmax joint uses (reduction, reduce, _loss.blank)."""

        def __init__(self, blank):
            super().__init__()
            self.reduction = "mean_batch"
            self._loss = types.SimpleNamespace(blank=blank)

        def forward(self, log_probs, **kwargs):
            return log_probs.new_zeros(log_probs.shape[0]) if self.reduction is None else log_probs.new_zeros(())

        def reduce(self, losses, target_lengths):     # the multi-softmax joint passes a list of sub-batch losses
            losses = losses if isinstance(losses, (list, tuple)) else [losses]
            return sum(x.sum() for x in losses) * 0
    model.loss = NoRNNTLoss(model.loss._loss.blank if hasattr(model.loss, "_loss") else 0)
    # the fork computes the RNN-T loss inside the joint ("fused"); its unfused validation path is broken
    # (log_probs used before assignment), so keep fusion and swap the loss inside the joint
    if model.joint.fuse_loss_wer:
        model.joint.set_loss(model.loss)
    with open_dict(model.cfg):
        model.cfg.aux_ctc.ctc_loss_weight = 1.0
    model.ctc_loss_weight = 1.0
    model.cur_decoder = "ctc"
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_optimization(model.cfg.optim)

    ckpt = pl.callbacks.ModelCheckpoint(dirpath=str(out / "ckpt"), monitor="val_wer", mode="min", save_top_k=1)
    trainer = pl.Trainer(devices=1, accelerator="gpu", precision="16-mixed", max_steps=args.max_steps,
                         val_check_interval=args.val_every, check_val_every_n_epoch=None, log_every_n_steps=50,
                         callbacks=[ckpt], enable_checkpointing=True, logger=False, gradient_clip_val=1.0)
    trainer.validate(model)                      # zero-shot dev score, logged before any training
    trainer.fit(model)
    # load_from_checkpoint cannot rebuild the multilingual tokenizer from a Lightning checkpoint: load the best
    # weights into the in-memory model instead, then save a regular .nemo
    if ckpt.best_model_path:
        model.load_state_dict(torch.load(ckpt.best_model_path, map_location="cpu")["state_dict"])
    model.save_to(str(out / "indicconformer_bn_120m_ft.nemo"))
    (out / "train_config.json").write_text(json.dumps({**vars(args), "train_clips": len(train), "tts_clips": len(tts),
                                                       "best_ckpt": ckpt.best_model_path,
                                                       "best_val_wer": float(ckpt.best_model_score or -1)}, indent=2))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
