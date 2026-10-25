"""Reusable Bengali ASR wrapper used by the Task 2 evaluations and the Task 3 pipeline.

Two fixes from the Task 2 error analysis are built in:
  1. Output budget: Whisper's byte-level tokenizer spends ~2 tokens per Bengali character,
     so the common max_new_tokens=128 cut long sentences mid-character ("missing words at the end").
     Default here is 440 (the decoder holds 448 incl. start tokens).
  2. VAD gate (optional): Whisper invents text for audio with no speech (hallucination).
     Silero VAD checks each clip first: no speech -> return "" without calling Whisper.
     Optional crop to the detected speech was tested and REJECTED: it cut real words at
     segment edges (consult B16 lost "স্যার স্টুল টেস্ট"), so by default the clip is kept whole.
"""
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000
_OPENAI_BY_WIDTH = {384: "openai/whisper-tiny", 512: "openai/whisper-base", 768: "openai/whisper-small",
                    1024: "openai/whisper-medium", 1280: "openai/whisper-large-v2"}


def fix_generation_config(model):
    """Some community checkpoints (e.g. the BengaliAI tugstugi models) ship an old-style generation
    config without language ids, which newer transformers rejects when language= is passed.
    The vocabulary is OpenAI's, so borrow the generation config of the same-size OpenAI model."""
    from transformers import GenerationConfig
    if getattr(model.generation_config, "lang_to_id", None):
        return model
    base = _OPENAI_BY_WIDTH[model.config.d_model]
    if model.config.num_mel_bins == 128:
        base = "openai/whisper-large-v3"
    model.generation_config = GenerationConfig.from_pretrained(base)
    return model


class BengaliASR:
    def __init__(self, model_path, max_new_tokens=440, num_beams=1, vad_gate=False,
                 vad_min_speech_ms=100, vad_pad_ms=150, vad_crop=False, prompt=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
        fix_generation_config(self.model).to(self.device).eval()
        self.gen = {"language": "bn", "task": "transcribe", "max_new_tokens": max_new_tokens, "num_beams": num_beams}
        if prompt:
            prompt_ids = self.processor.get_prompt_ids(prompt)
            self.gen["prompt_ids"] = torch.tensor(prompt_ids).to(self.device)
            # the decoder holds 448 tokens in total: prompt + start tokens + output
            self.gen["max_new_tokens"] = min(max_new_tokens, 448 - len(prompt_ids) - 4)
        self.vad_gate, self.vad_min_speech_ms, self.vad_pad_ms = vad_gate, vad_min_speech_ms, vad_pad_ms
        self.vad_crop = vad_crop
        self._vad = None

    def speech_crop(self, clip):
        """Return the clip cropped to detected speech, or None if there is no speech."""
        from silero_vad import get_speech_timestamps, load_silero_vad
        if self._vad is None:
            self._vad = load_silero_vad()
        ts = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(clip)), self._vad, sampling_rate=SR,
                                   min_speech_duration_ms=self.vad_min_speech_ms)
        if not ts:
            return None
        pad = int(self.vad_pad_ms / 1000 * SR)
        return clip[max(0, ts[0]["start"] - pad): min(len(clip), ts[-1]["end"] + pad)]

    def transcribe(self, clips, batch=8):
        """clips: list of float32 numpy arrays at 16 kHz -> list of strings."""
        texts = [""] * len(clips)
        todo = []
        for i, c in enumerate(clips):
            if self.vad_gate:
                cropped = self.speech_crop(c)
                if cropped is None:
                    continue                      # no speech: don't let Whisper invent text
                if self.vad_crop:
                    c = cropped
            todo.append((i, c))
        for k in range(0, len(todo), batch):
            idx, chunk = zip(*todo[k:k + batch])
            feats = self.processor.feature_extractor(list(chunk), sampling_rate=SR, return_tensors="pt",
                                                     return_attention_mask=True)
            with torch.no_grad():
                ids = self.model.generate(feats.input_features.to(self.device, self.dtype),
                                          attention_mask=feats.attention_mask.to(self.device), **self.gen)
            for i, t in zip(idx, self.processor.batch_decode(ids, skip_special_tokens=True)):
                texts[i] = t.strip()
        return texts
