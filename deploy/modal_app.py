"""AIMScribe on Modal: the three assessment tasks as a web page and a REST API.

  Task 1  POST /diarize     audio -> speaker segments: pyannote community-1 (final), pyannote 3.1 (baseline) or
                            3.1 + my segment re-clustering; with a reference RTTM also DER, miss / false alarm /
                            confusion, DER inside / outside overlap. 3.1 runs in its own container with pyannote 3.3.2.
  Task 2  POST /asr         audio -> Bengali text (IndicConformer 600M, my fine-tuned whisper-small, or zero-shot
                            whisper-small); with a reference text also WER / CER
  Task 3  POST /transcribe  audio -> speaker-attributed, role-labelled transcript (task3_pipeline/pipeline.py)

The code of the three tasks runs unchanged; models load once per container. Diarization and Whisper run on a T4 GPU,
IndicConformer on CPU (onnxruntime). The container starts on the first request and stops after SCALEDOWN_S idle seconds.

Setup (once):
  pip install modal && modal setup
  Modal dashboard -> Secrets -> new secret "aimscribe" with
      HF_TOKEN  Hugging Face read token whose account accepted the terms of the gated models
      API_KEY   requests must send it in the X-API-Key header (leave empty for an open endpoint)
Deploy:
  modal deploy deploy/modal_app.py
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_DIR = "/root/aimscribe"
MAX_AUDIO_S = 600          # longest accepted recording
SCALEDOWN_S = 300          # idle seconds before the GPU container stops
CHUNK_S = 25.0             # Task 2: longer audio is cut at pauses into chunks of at most this length
CONSULTS = ["consult_A", "consult_B"]
WHISPER = {"whisper-small-ft": "sahilfarib/aimscribe-whisper-small-bn", "whisper-small": "openai/whisper-small"}
ASR_MODELS = ["indicconformer-600m", *WHISPER]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(   # same versions as the local pyannote 4 environment that produced the reported results
        "torch==2.14.1", "torchaudio==2.11.0", "torchcodec==0.17.0", "pyannote.audio==4.0.7", "pyannote.metrics==4.2.0",
        "onnxruntime==1.20.1", "silero-vad==5.1.2", "transformers==4.46.3", "numpy==2.5.4", "scipy==1.18.1",
        "soundfile==0.14.0", "jiwer==3.0.5", "huggingface_hub==0.36.2", "python-dotenv", "fastapi[standard]",
        "python-multipart",
    )
    .env({"AIMSCRIBE_DATA": "/models", "HF_HOME": "/models/hf"})
    .add_local_file(REPO / "data" / "common.py", f"{APP_DIR}/data/common.py")
    .add_local_file(REPO / "deploy" / "page.html", f"{APP_DIR}/page.html")
    .add_local_dir(REPO / "task1_diarization", f"{APP_DIR}/task1_diarization", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "task2_asr", f"{APP_DIR}/task2_asr", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "task3_pipeline", f"{APP_DIR}/task3_pipeline", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "data" / "samples", f"{APP_DIR}/samples", ignore=["**/*.png", "**/README.md"])
)
for _name in CONSULTS:
    image = image.add_local_file(REPO / "data" / "consultation" / "audio" / f"{_name}.wav", f"{APP_DIR}/consult/{_name}.wav")

# Task 1 baseline and my re-clustering ran in the main environment (pyannote.audio 3.3.2); pyannote 4 runs the 3.1
# pipeline differently, so they get their own container with the original versions (requirements.txt, constraints.txt)
baseline_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_requirements(str(REPO / "deploy" / "requirements-baseline.txt"))   # exact laptop versions
    .env({"AIMSCRIBE_DATA": "/models", "HF_HOME": "/models/hf"})
    .add_local_file(REPO / "data" / "common.py", f"{APP_DIR}/data/common.py")
    .add_local_dir(REPO / "task1_diarization", f"{APP_DIR}/task1_diarization", ignore=["**/__pycache__/**"])
)
models = modal.Volume.from_name("aimscribe-models", create_if_missing=True)
app = modal.App("aimscribe", image=image)


def parse_rttm(text, uri="audio"):
    """RTTM text -> pyannote Annotation (one track per line)."""
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uri)
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            ann[Segment(float(p[3]), float(p[3]) + float(p[4])), len(ann)] = p[7]
    return ann


@app.cls(image=baseline_image, gpu="T4", volumes={"/models": models}, secrets=[modal.Secret.from_name("aimscribe")],
         scaledown_window=SCALEDOWN_S, timeout=900, max_containers=1)
class Baseline:
    """pyannote 3.1 (Task 1 baseline) and pyannote 3.1 + my segment re-clustering, as in diarize.py / recluster.py."""

    @modal.enter()
    def load(self):
        import os
        import sys

        import torch
        from pyannote.audio import Inference, Model, Pipeline
        sys.path[:0] = [f"{APP_DIR}/data", f"{APP_DIR}/task1_diarization"]
        from recluster import EMB_MODEL, recluster
        self.recluster = recluster
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = os.environ["HF_TOKEN"]
        self.sd = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token).to(device)
        self.emb = Inference(Model.from_pretrained(EMB_MODEL, use_auth_token=token), window="whole", device=device)
        models.commit()

    @modal.method()
    def diarize(self, audio, num_speakers=None, recluster=False):
        """audio: 16 kHz mono float32 -> RTTM text. Re-clustering reads the 3.1 output back from RTTM, like recluster.py."""
        import io
        import tempfile

        import torch
        from pyannote.database.util import load_rttm
        inp = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000, "uri": "audio"}
        hyp = self.sd(inp, **({"num_speakers": num_speakers} if num_speakers else {}))
        if recluster:
            with tempfile.NamedTemporaryFile("w", suffix=".rttm", delete=False) as f:
                hyp.write_rttm(f)
            hyp = load_rttm(f.name).get("audio") or hyp
            hyp = self.recluster(hyp, audio, 16000, self.emb, 0.70, num_speakers, min_dur=1.0)
        out = io.StringIO()
        hyp.write_rttm(out)
        return out.getvalue()


@app.cls(gpu="T4", cpu=4, memory=16384, volumes={"/models": models}, secrets=[modal.Secret.from_name("aimscribe")],
         scaledown_window=SCALEDOWN_S, timeout=1200, max_containers=1)
@modal.concurrent(max_inputs=4)
class AIMScribe:
    @modal.enter()
    def load(self):
        import os
        import sys
        import threading

        from huggingface_hub import snapshot_download

        asr_dir = "/models/models/indic-conformer-600m"
        if not os.path.exists(f"{asr_dir}/model_onnx.py"):     # first start only: weights persist in the volume
            snapshot_download("ai4bharat/indic-conformer-600m-multilingual", local_dir=asr_dir,
                              token=os.environ["HF_TOKEN"])
        sys.path.insert(0, f"{APP_DIR}/task3_pipeline")
        import pipeline
        self.pl = pipeline
        self.pipe = pipeline.Pipeline(asr_dir)                  # Task 3 = community-1 + IndicConformer 600M + roles
        self.whisper = {}                                       # Whisper models are loaded on first use
        models.commit()
        self.lock = threading.Lock()                            # one job at a time on the single GPU

    # ------------------------------------------------------------------ Task 1
    def diarize(self, path, system, num_speakers=None, reference=None):
        import io
        import time

        import torch
        t0 = time.time()
        audio = self.pl.load_audio(path)
        if system == "community1":
            inp = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": self.pl.SR, "uri": "audio"}
            with self.lock:
                out = self.pipe.sd(inp, **({"num_speakers": num_speakers} if num_speakers else {}))
            hyp = getattr(out, "speaker_diarization", out)
            rttm = io.StringIO()
            hyp.write_rttm(rttm)
            rttm = rttm.getvalue()
        else:                                                   # pyannote 3.1 (+ re-clustering) in its own container
            rttm = Baseline().diarize.remote(audio, num_speakers, recluster=system == "pyannote31_recluster")
            hyp = parse_rttm(rttm)
        segments = [{"speaker": spk, "start": round(seg.start, 2), "end": round(seg.end, 2)}
                    for seg, _, spk in sorted(hyp.itertracks(yield_label=True), key=lambda x: x[0].start)]
        result = {"system": system, "speakers": len(hyp.labels()), "segments": segments, "rttm": rttm,
                  "duration_s": round(len(audio) / self.pl.SR, 2), "processing_s": round(time.time() - t0, 1)}
        if reference:
            result["evaluation"] = self.der(reference, hyp)
        return result

    @staticmethod
    def der(reference_rttm, hyp):
        """DER exactly as task1_diarization/evaluate.py: collar 0.25 s, overlap scored; also inside / outside the
        regions where 2+ reference speakers talk."""
        from pyannote.core import Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate
        ref = parse_rttm(reference_rttm)
        if not len(ref):
            raise ValueError("the reference RTTM has no SPEAKER lines")
        hyp = hyp.rename_tracks()
        hyp.uri = "audio"
        d = DiarizationErrorRate(collar=0.25, skip_overlap=False)(ref, hyp, detailed=True)
        ov = ref.get_overlap()
        end = max(ref.get_timeline().extent().end, hyp.get_timeline().extent().end if len(hyp) else 0) + 1
        rest = Timeline([Segment(0, end)], uri="audio").extrude(ov)
        pct = lambda x: round(100 * x, 2)
        return {"DER%": pct(d["diarization error rate"]), "miss%": pct(d["missed detection"] / d["total"]),
                "FA%": pct(d["false alarm"] / d["total"]), "conf%": pct(d["confusion"] / d["total"]),
                "DER_overlap%": pct(DiarizationErrorRate(collar=0.25)(ref, hyp, uem=ov)) if ov else None,
                "DER_rest%": pct(DiarizationErrorRate(collar=0.25)(ref, hyp, uem=rest)),
                "ref_speakers": len(ref.labels()), "hyp_speakers": len(hyp.labels()), "collar_s": 0.25}

    # ------------------------------------------------------------------ Task 2
    def asr_model(self, name, max_new_tokens, vad_gate):
        if name == "indicconformer-600m":
            return self.pipe.asr                                # VAD gate on, as in the pipeline
        if name not in self.whisper:
            self.whisper[name] = self.pl.BengaliASR(WHISPER[name])
            models.commit()
        m = self.whisper[name]
        m.gen["max_new_tokens"], m.vad_gate = max_new_tokens, vad_gate
        return m

    def chunks(self, audio):
        """<= CHUNK_S: one clip. Longer: cut at pauses (Silero VAD) into chunks of at most CHUNK_S."""
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        sr, n = self.pl.SR, int(CHUNK_S * self.pl.SR)
        if len(audio) <= n:
            return [(0, len(audio))]
        speech = get_speech_timestamps(torch.from_numpy(audio), load_silero_vad(), sampling_rate=sr,
                                       min_silence_duration_ms=300)
        out = []
        for s in speech:
            if out and s["end"] - out[-1][0] <= n:
                out[-1] = (out[-1][0], s["end"])
            else:
                for a in range(s["start"], s["end"], n):        # a single speech region longer than CHUNK_S
                    out.append((a, min(a + n, s["end"])))
        return out or [(0, len(audio))]

    def asr(self, path, model, max_new_tokens=440, vad_gate=True, reference=None):
        import time

        import jiwer
        from common import normalize_bn
        t0 = time.time()
        audio = self.pl.load_audio(path)
        spans = self.chunks(audio)
        with self.lock:
            texts = self.asr_model(model, max_new_tokens, vad_gate).transcribe([audio[a:b] for a, b in spans])
        sr = self.pl.SR
        chunks = [{"start": round(a / sr, 2), "end": round(b / sr, 2), "text": normalize_bn(t)}
                  for (a, b), t in zip(spans, texts)]
        text = " ".join(c["text"] for c in chunks if c["text"])
        whisper = model in WHISPER
        result = {"model": model, "text": text, "chunks": chunks, "duration_s": round(len(audio) / sr, 2),
                  "processing_s": round(time.time() - t0, 1),
                  "settings": {"max_new_tokens": max_new_tokens if whisper else None,
                               "vad_gate": vad_gate if whisper else True}}
        ref = normalize_bn(reference or "")
        if ref:                                                 # no Bengali text left after normalising: no score
            result["evaluation"] = {"WER%": round(100 * jiwer.wer(ref, text or " "), 2),
                                    "CER%": round(100 * jiwer.cer(ref, text or " "), 2), "reference": ref}
        return result

    # ------------------------------------------------------------------ Task 3
    def transcribe(self, path, num_speakers=None):
        import io
        with self.lock:
            self.pipe.num_speakers = num_speakers
            r = self.pipe.run(path)
        rttm = io.StringIO()
        r["diarization"].write_rttm(rttm)
        return {
            "transcript": self.pl.format_transcript(r),
            "turns": [{"role": t["role"], "speaker": t["speaker"], "start": round(t["start"], 2),
                       "end": round(t["end"], 2), "overlap": t["overlap"], "text": t["text"]} for t in r["turns"]],
            "roles": r["roles"],
            "duration_s": r["duration_s"],
            "processing_s": r["timing_s"],
            "rttm": rttm.getvalue(),
        }

    # ------------------------------------------------------------------ web
    @modal.asgi_app()
    def web(self):
        import csv
        import os
        import tempfile

        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse

        api = FastAPI(title="AIMScribe: Bengali speech AI assessment",
                      description="Task 1 speaker diarization, Task 2 Bengali ASR, Task 3 role-labelled transcript of "
                                  "a doctor-patient consultation. Send the access key in the X-API-Key header.")
        key = os.environ.get("API_KEY", "")
        samples = Path(f"{APP_DIR}/samples")

        def check_key(given):
            if key and given != key:
                raise HTTPException(401, "missing or wrong X-API-Key header")

        def run(file, fn):
            """Save the upload, check its length, run fn(path), always clean up."""
            suffix = Path(file.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(file.file.read())
                path = f.name
            try:
                seconds = len(self.pl.load_audio(path)) / self.pl.SR
                if seconds > MAX_AUDIO_S:
                    raise HTTPException(413, f"recording is {seconds:.0f} s; the limit is {MAX_AUDIO_S} s")
                return fn(path)
            except HTTPException:
                raise
            except Exception as e:                              # unreadable audio, bad reference, ...
                raise HTTPException(400, f"could not process the file: {e}")
            finally:
                os.unlink(path)

        @api.get("/", response_class=HTMLResponse, include_in_schema=False)
        def page():
            html = Path(f"{APP_DIR}/page.html").read_text(encoding="utf-8")
            return html.replace("__KEY_REQUIRED__", "true" if key else "false")

        @api.get("/health")
        def health():
            return {"status": "ok"}

        @api.post("/diarize", summary="Task 1: speaker diarization (+ DER with a reference RTTM)")
        def diarize(file: UploadFile = File(..., description="audio file: wav, mp3, m4a, ..."),
                    system: str = Form("community1", description="community1 (final) | pyannote31 (baseline) | "
                                                                 "pyannote31_recluster (3.1 + my re-clustering)"),
                    num_speakers: int | None = Form(None, description="number of speakers, if known"),
                    reference: UploadFile | None = File(None, description="reference RTTM, optional"),
                    x_api_key: str | None = Header(None)):
            check_key(x_api_key)
            if system not in ("community1", "pyannote31", "pyannote31_recluster"):
                raise HTTPException(422, "system must be community1, pyannote31 or pyannote31_recluster")
            ref = reference.file.read().decode("utf-8") if reference else None
            return run(file, lambda p: self.diarize(p, system, num_speakers, ref))

        @api.post("/asr", summary="Task 2: Bengali speech recognition (+ WER / CER with a reference text)")
        def asr(file: UploadFile = File(..., description="audio file: wav, mp3, m4a, ..."),
                model: str = Form("indicconformer-600m", description=" | ".join(ASR_MODELS)),
                max_new_tokens: int = Form(440, description="Whisper only: output token budget, 128 or 440"),
                vad_gate: bool = Form(True, description="Whisper only: skip audio without detected speech"),
                reference: str | None = Form(None, description="reference transcript, optional"),
                x_api_key: str | None = Header(None)):
            check_key(x_api_key)
            if model not in ASR_MODELS:
                raise HTTPException(422, f"model must be one of {ASR_MODELS}")
            if max_new_tokens not in (128, 440):
                raise HTTPException(422, "max_new_tokens must be 128 or 440")
            return run(file, lambda p: self.asr(p, model, max_new_tokens, vad_gate, reference))

        @api.post("/transcribe", summary="Task 3: speaker-attributed, role-labelled transcript")
        def transcribe(file: UploadFile = File(..., description="audio file: wav, mp3, m4a, ..."),
                       num_speakers: int | None = Form(None, description="number of speakers, if known"),
                       x_api_key: str | None = Header(None)):
            check_key(x_api_key)
            return run(file, lambda p: self.transcribe(p, num_speakers))

        # ---- samples: the annotated data shipped with the submission ----
        @api.get("/samples", summary="Sample audio with references and the values reported in the submission")
        def sample_list():
            diar = {}
            with open(samples / "diarization" / "samples.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    d = diar.setdefault(r["sample"], {"name": r["sample"], "set": r["set"],
                                                      "ref_speakers": int(r["ref_speakers"]), "reported_DER%": {}})
                    d["reported_DER%"][r["system"]] = float(r["DER%"])
            with open(samples / "asr" / "transcripts.csv", encoding="utf-8") as f:
                asr_rows = [{"file": r["path"], "text": r["text"], "duration_s": float(r["duration"])}
                            for r in csv.DictReader(f)]
            return {"diarization": list(diar.values()), "asr": asr_rows, "consultations": CONSULTS}

        def safe(name):
            if not name or "/" in name or "\\" in name or ".." in name:
                raise HTTPException(404, "unknown sample")
            return name

        @api.get("/samples/diarization/{name}/audio.wav", include_in_schema=False)
        def sample_diar_audio(name: str):
            p = samples / "diarization" / safe(name) / "audio.wav"
            if not p.exists():
                raise HTTPException(404, "unknown sample")
            return FileResponse(p, media_type="audio/wav")

        @api.get("/samples/diarization/{name}/reference.rttm", include_in_schema=False)
        def sample_diar_ref(name: str):
            p = samples / "diarization" / safe(name) / "reference.rttm"
            if not p.exists():
                raise HTTPException(404, "unknown sample")
            return FileResponse(p, media_type="text/plain")

        @api.get("/samples/asr/{file}", include_in_schema=False)
        def sample_asr(file: str):
            p = samples / "asr" / safe(file)
            if not file.endswith(".wav") or not p.exists():
                raise HTTPException(404, "unknown sample")
            return FileResponse(p, media_type="audio/wav")

        @api.get("/samples/{name}.wav", include_in_schema=False)
        def sample_consult(name: str):
            if name not in CONSULTS:
                raise HTTPException(404, "unknown sample")
            return FileResponse(f"{APP_DIR}/consult/{name}.wav", media_type="audio/wav")

        return api
