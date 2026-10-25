"""Non-Whisper ASR candidates, wrapped with the same .transcribe(clips) interface as BengaliASR,
so eval_consult.py scores them on exactly the same cuts and normalisation.

  IndicConformer  AI4Bharat, 600M, trained on 22 Indic languages (incl. Bengali), CTC / RNN-T decoding.
                  A speech specialist: no language model on top, so it can't "invent" fluent text.
  Gemma 3n        Google's open multimodal LLM (audio in, text out). General-purpose, not an ASR model;
                  prompted to transcribe. Bengali speech quality is not published, hence this test.
Both are gated on Hugging Face (accept the terms with the account of HF_TOKEN).
"""
import numpy as np
import torch

SR = 16000


class IndicConformerASR:
    def __init__(self, model_id="ai4bharat/indic-conformer-600m-multilingual", decoding="rnnt", lang="bn",
                 vad_gate=False, vad_min_speech_ms=100):
        """model_id: hub id, or a folder from snapshot_download(local_dir=...). The ONNX weights refuse to load
        through the HF cache's symlinked blobs on Colab, so a plain local folder is the reliable route."""
        import os
        if os.path.isdir(model_id):
            import importlib.util
            spec = importlib.util.spec_from_file_location("indic_model_onnx", os.path.join(model_id, "model_onnx.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.model = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=model_id))
            # The model's TorchScript feature extractor (STFT, complex numbers) goes to CUDA by default, and
            # torch 2.5's GPU fuser fails to compile it. The network itself runs on onnxruntime (CPU), so
            # computing the spectrogram on CPU costs little and works on every torch version.
            self.model.d = torch.device("cpu")
            self.model.models["preprocessor"] = torch.jit.load(os.path.join(model_id, "assets", "preprocessor.ts"),
                                                               map_location="cpu")
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.decoding, self.lang = decoding, lang
        self.vad_gate, self.vad_min_speech_ms, self._vad = vad_gate, vad_min_speech_ms, None

    def has_speech(self, clip):
        """Same Silero VAD gate as BengaliASR: no speech -> skip the model."""
        from silero_vad import get_speech_timestamps, load_silero_vad
        if self._vad is None:
            self._vad = load_silero_vad()
        return bool(get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(clip)), self._vad, sampling_rate=SR,
                                          min_speech_duration_ms=self.vad_min_speech_ms))

    def transcribe(self, clips, batch=None):
        out = []
        for c in clips:
            if self.vad_gate and not self.has_speech(c):
                out.append("")
                continue
            wav = torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                out.append(str(self.model(wav, self.lang, self.decoding)).strip())
        return out


class IndicConformerNemoASR:
    """AI4Bharat IndicConformer, Bengali-only, 120M (hybrid CTC / RNN-T), a regular .nemo checkpoint.
    Unlike the 600M ONNX release it can be FINE-TUNED. Needs AI4Bharat's NeMo fork (branch nemo-v2): stock NeMo
    doesn't know its multi-softmax decoder. Runs in its own environment (see notebooks)."""

    def __init__(self, nemo_path, decoding="rnnt", lang="bn", vad_gate=False, vad_min_speech_ms=100):
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.restore_from(nemo_path).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.cur_decoder = decoding
        self.lang = lang
        self.vad_gate, self.vad_min_speech_ms, self._vad = vad_gate, vad_min_speech_ms, None

    has_speech = IndicConformerASR.has_speech

    def transcribe(self, clips, batch=8):
        import os
        import tempfile
        import soundfile as sf
        keep = [i for i, c in enumerate(clips) if not self.vad_gate or self.has_speech(c)]
        out = [""] * len(clips)
        with tempfile.TemporaryDirectory() as d:   # this NeMo version transcribes files, not arrays
            paths = []
            for i in keep:
                paths.append(os.path.join(d, f"{i}.wav"))
                sf.write(paths[-1], np.asarray(clips[i], dtype=np.float32), SR)
            with torch.no_grad():
                hyps = self.model.transcribe(paths, batch_size=batch, language_id=self.lang)
        if isinstance(hyps, tuple):                # RNN-T returns (best, all)
            hyps = hyps[0]
        for i, h in zip(keep, hyps):
            out[i] = str(getattr(h, "text", h)).strip()
        return out


class Gemma3nASR:
    PROMPT = "Transcribe this Bengali audio exactly, in Bengali script. Output only the transcript."

    def __init__(self, model_id="google/gemma-3n-E2B-it", max_new_tokens=256):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        # T4 has no bfloat16; float16 halves memory (E2B ~10 GB)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                 device_map="cuda").eval()
        self.max_new_tokens = max_new_tokens

    def transcribe(self, clips, batch=None):
        out = []
        for c in clips:
            msgs = [{"role": "user", "content": [{"type": "audio", "audio": np.asarray(c, dtype=np.float32)},
                                                  {"type": "text", "text": self.PROMPT}]}]
            inputs = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt").to(self.model.device)
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
            with torch.no_grad():
                ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            out.append(self.processor.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        return out
