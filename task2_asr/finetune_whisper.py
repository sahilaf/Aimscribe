"""Task 2: fine-tune Whisper on the small Bengali training set (run on a Colab T4).

Choices (and why):
- whisper-small (244M params): multilingual incl. Bengali; fits a 15 GB T4 with batch 8 x 2 accumulation
  in fp16 (batch 16 in one go ran out of memory).
  whisper-medium/large would need gradient tricks and much longer training.
- Full fine-tuning, lr 1e-5, short warmup: standard recipe for Whisper on small data;
  a higher lr makes it forget its multilingual knowledge (catastrophic forgetting).
- Dev split = a few TRAINING speakers held out (speaker-disjoint). The best checkpoint is chosen on dev WER;
  the test set is never seen during training or model selection.
- Language/task tokens fixed to Bengali + transcribe, so the model can't drift into another script.
- fp16 and only-model checkpoints on fast local disk; the best model is copied to Drive at the end.

- --lora (for the 769M tugstugi Bengali whisper-medium): full fine-tuning would need ~15 GB with AdamW
  and doesn't fit the T4. LoRA freezes the original weights and trains small low-rank adapters on the
  attention projections (~1% of parameters), which also limits forgetting. Adapters are merged back
  at the end, so the saved model loads like any Whisper checkpoint.

Usage (Colab):
  python task2_asr/finetune_whisper.py --run ft_small --max_steps 800
  python task2_asr/finetune_whisper.py --run ft_medium_lora --base bengaliAI/tugstugi_bengaliai-asr_whisper-medium \
      --lora --match_tugstugi --lr 1e-4 --batch 4 --grad_accum 4 --gen_max_len 225
"""
import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import jiwer
import pandas as pd
import soundfile as sf
import torch
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from asr import fix_generation_config  # noqa: E402
from common import DATA_ROOT, REPO, normalize_bn  # noqa: E402


# য় ড় ঢ় can be written as one code point or as letter + nukta (়). They look identical; NFC picks the
# two-code-point form, but the tugstugi model was trained on (and outputs) the one-code-point form.
# Teaching it the other form would cost loss on words it already gets right. Evaluation is unaffected
# (normalize_bn applies NFC to both sides).
PRECOMPOSE = {"য়": "য়", "ড়": "ড়", "ঢ়": "ঢ়"}


def precompose(text):
    for k, v in PRECOMPOSE.items():
        text = text.replace(k, v)
    return text


class AudioTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, text_fn=None):
        self.df, self.processor = df.reset_index(drop=True), processor
        if text_fn:
            self.df["text"] = self.df["text"].map(text_fn)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        audio, sr = sf.read(DATA_ROOT / row["path"], dtype="float32")
        feats = self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        labels = self.processor.tokenizer(row["text"]).input_ids     # <|startoftranscript|><|bn|><|transcribe|>... <|endoftext|>
        return {"input_features": feats, "labels": labels}


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int
    mask_prefix: bool = False

    def __call__(self, batch):
        feats = self.processor.feature_extractor.pad(
            [{"input_features": b["input_features"]} for b in batch], return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)   # padding ignored by the loss
        if self.mask_prefix:
            # <|startoftranscript|><|bn|><|transcribe|><|notimestamps|> are forced at inference, so the model
            # only needs to predict the text. Feed them as decoder input but leave them out of the loss
            # (the tugstugi model was never trained to predict them: ~65 nats per sentence otherwise).
            feats["decoder_input_ids"] = labels["input_ids"][:, :-1]
            ids = ids[:, 1:].clone()
            ids[:, :3] = -100
        elif (ids[:, 0] == self.decoder_start_token_id).all():   # the model adds the start token itself
            ids = ids[:, 1:]
        feats["labels"] = ids
        return feats


def speaker_split(df, dev_speakers, seed):
    spk = sorted(df["speaker"].unique())
    random.Random(seed).shuffle(spk)
    dev = set(spk[:dev_speakers])
    return df[~df.speaker.isin(dev)], df[df.speaker.isin(dev)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--train_csv", default=str(REPO / "data" / "asr" / "train.csv"))
    ap.add_argument("--dev_speakers", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2, help="effective batch = batch x grad_accum (16 OOMs a T4 in one go)")
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--spec_augment", action="store_true", help="mask time/frequency bands (regularisation)")
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--gen_max_len", type=int, default=128, help="decode length for dev evaluation")
    ap.add_argument("--match_tugstugi", action="store_true",
                    help="label format of the tugstugi models: one-code-point য়/ড়/ঢ়, prefix tokens not in the loss")
    ap.add_argument("--work_dir", default="/content/ckpt")
    ap.add_argument("--save_to", default="/content/checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    processor = WhisperProcessor.from_pretrained(args.base, language="bengali", task="transcribe")
    model = fix_generation_config(WhisperForConditionalGeneration.from_pretrained(args.base))
    model.generation_config.language = "bengali"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_feature_prob = 0.05
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
        model.print_trainable_parameters()

    train_df, dev_df = speaker_split(pd.read_csv(args.train_csv), args.dev_speakers, args.seed)
    print(f"train: {len(train_df)} utts / {train_df.duration.sum() / 60:.1f} min / {train_df.speaker.nunique()} spk | "
          f"dev: {len(dev_df)} utts / {dev_df.duration.sum() / 60:.1f} min / {dev_df.speaker.nunique()} spk")

    def compute_metrics(pred):
        ids = pred.label_ids
        ids[ids == -100] = processor.tokenizer.pad_token_id
        hyp = [normalize_bn(t) for t in processor.batch_decode(pred.predictions, skip_special_tokens=True)]
        ref = [normalize_bn(t) for t in processor.batch_decode(ids, skip_special_tokens=True)]
        return {"wer": 100 * jiwer.wer(ref, hyp), "cer": 100 * jiwer.cer(ref, hyp)}

    targs = Seq2SeqTrainingArguments(
        output_dir=f"{args.work_dir}/{args.run}",
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        max_steps=args.max_steps,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_only_model=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.gen_max_len,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=AudioTextDataset(train_df, processor, precompose if args.match_tugstugi else None),
        eval_dataset=AudioTextDataset(dev_df, processor, precompose if args.match_tugstugi else None),
        data_collator=Collator(processor, model.config.decoder_start_token_id, mask_prefix=args.match_tugstugi),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    zero_shot = trainer.evaluate() if args.lora else None     # dev score before any training, for reference
    trainer.train()

    final = Path(args.save_to) / args.run
    if final.exists():
        shutil.rmtree(final)
    if args.lora:
        trainer.model.merge_and_unload().save_pretrained(str(final))   # plain Whisper checkpoint
    else:
        trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    out = REPO / "results" / "task2" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (out / "train_config.json").write_text(json.dumps({**vars(args), "best_checkpoint": trainer.state.best_model_checkpoint,
                                                       "best_dev_wer": trainer.state.best_metric,
                                                       "zero_shot_dev": zero_shot,
                                                       "train_utts": len(train_df), "dev_utts": len(dev_df)}, indent=2))
    print(f"best dev WER {trainer.state.best_metric:.2f} at {trainer.state.best_model_checkpoint} -> saved to {final}")


if __name__ == "__main__":
    main()
