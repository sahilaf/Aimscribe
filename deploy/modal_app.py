"""AIMScribe pipeline as a web service on Modal: Bengali consultation audio -> role-labelled transcript.

Runs task3_pipeline/pipeline.py unchanged (community-1 diarization on a T4 GPU, Silero VAD + IndicConformer 600M on
CPU, role rules). The container starts on the first request and stops after SCALEDOWN_S idle seconds.

Setup (once):
  pip install modal && modal setup
  Modal dashboard -> Secrets -> new secret "aimscribe" with
      HF_TOKEN  Hugging Face read token whose account accepted the terms of
                pyannote/speaker-diarization-community-1 and ai4bharat/indic-conformer-600m-multilingual
      API_KEY   any string; requests must send it in the X-API-Key header (leave empty for an open endpoint)
Deploy:
  modal deploy deploy/modal_app.py

Endpoints:
  GET  /             upload page
  POST /transcribe   multipart: file (any audio format), num_speakers (optional) -> JSON
  GET  /docs         interactive API documentation
  GET  /health
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_DIR = "/root/aimscribe"
MAX_AUDIO_S = 600          # longest accepted recording
SCALEDOWN_S = 300          # idle seconds before the GPU container stops
SAMPLES = ["consult_A", "consult_B"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(   # same versions as the local pyannote 4 environment that produced the reported results
        "torch==2.14.1", "torchaudio==2.11.0", "torchcodec==0.17.0", "pyannote.audio==4.0.7", "onnxruntime==1.20.1",
        "silero-vad==5.1.2", "transformers==4.46.3", "numpy==2.5.4", "scipy==1.18.1", "soundfile==0.14.0",
        "huggingface_hub==0.36.2", "python-dotenv", "fastapi[standard]", "python-multipart",
    )
    .env({"AIMSCRIBE_DATA": "/models", "HF_HOME": "/models/hf"})
    .add_local_file(REPO / "data" / "common.py", f"{APP_DIR}/data/common.py")
    .add_local_dir(REPO / "task1_diarization", f"{APP_DIR}/task1_diarization", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "task2_asr", f"{APP_DIR}/task2_asr", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO / "task3_pipeline", f"{APP_DIR}/task3_pipeline", ignore=["**/__pycache__/**"])
)
for _name in SAMPLES:
    image = image.add_local_file(REPO / "data" / "consultation" / "audio" / f"{_name}.wav", f"{APP_DIR}/samples/{_name}.wav")
models = modal.Volume.from_name("aimscribe-models", create_if_missing=True)
app = modal.App("aimscribe", image=image)


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
        self.pipe = pipeline.Pipeline(asr_dir)
        models.commit()                                         # keep the downloaded pyannote weights too
        self.lock = threading.Lock()                            # one recording at a time on the single GPU

    def process(self, path, num_speakers=None):
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

    @modal.asgi_app()
    def web(self):
        import os
        import tempfile

        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
        from fastapi.responses import FileResponse, HTMLResponse

        api = FastAPI(title="AIMScribe: Bengali consultation transcription",
                      description="Upload a Bengali doctor-patient recording; get a speaker-attributed, "
                                  "role-labelled transcript (Doctor / Patient / Attendant / Other).")
        key = os.environ.get("API_KEY", "")

        def check_key(given):
            if key and given != key:
                raise HTTPException(401, "missing or wrong X-API-Key header")

        @api.get("/", response_class=HTMLResponse)
        def page():
            return PAGE.replace("{KEY_FIELD}", KEY_FIELD if key else "")

        @api.get("/health")
        def health():
            return {"status": "ok"}

        @api.get("/samples/{name}.wav")
        def sample(name: str):
            if name not in SAMPLES:
                raise HTTPException(404, "unknown sample")
            return FileResponse(f"{APP_DIR}/samples/{name}.wav", media_type="audio/wav")

        @api.post("/transcribe")
        def transcribe(file: UploadFile = File(..., description="audio file: wav, mp3, m4a, ..."),
                       num_speakers: int | None = Form(None, description="number of speakers, if known"),
                       x_api_key: str | None = Header(None)):
            check_key(x_api_key)
            suffix = Path(file.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(file.file.read())
                path = f.name
            try:
                seconds = len(self.pl.load_audio(path)) / self.pl.SR
                if seconds > MAX_AUDIO_S:
                    raise HTTPException(413, f"recording is {seconds:.0f} s; the limit is {MAX_AUDIO_S} s")
                return self.process(path, num_speakers)
            except HTTPException:
                raise
            except Exception as e:                              # unreadable audio etc.
                raise HTTPException(400, f"could not process the file: {e}")
            finally:
                os.unlink(path)

        return api


KEY_FIELD = '<label>Access key <input id="key" type="password" placeholder="X-API-Key"></label>'

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AIMScribe</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1d2733;background:#fff}
 h1{font-size:1.4rem;margin-bottom:.2rem} p{color:#4a5563;margin-top:.2rem}
 form{display:flex;flex-wrap:wrap;gap:.8rem;align-items:center;margin:1.2rem 0}
 input[type=number]{width:4rem} button{padding:.45rem 1rem;border:0;border-radius:6px;background:#1f4e79;color:#fff;cursor:pointer}
 button.alt{background:#e6edf3;color:#1f4e79} button:disabled{opacity:.5}
 pre{white-space:pre-wrap;background:#f5f7fa;border-radius:8px;padding:1rem;line-height:1.6;font-size:.95rem}
 #status{color:#4a5563} a{color:#1f4e79}
</style></head><body>
<h1>AIMScribe: Bengali consultation transcription</h1>
<p>Upload a doctor&ndash;patient recording (any audio format, up to 10 min). Output: who spoke when, their role, and
what they said. The first request after an idle period takes about a minute to start the server.</p>
<form id="f">
 <input id="file" type="file" accept="audio/*">
 <label>Speakers (optional) <input id="n" type="number" min="1" max="10"></label>
 {KEY_FIELD}
 <button id="go" type="submit">Transcribe</button>
 <button id="sample" class="alt" type="button">Try sample recording</button>
</form>
<div id="status"></div>
<pre id="out" hidden></pre>
<p id="dl" hidden><a id="json" download="transcript.json">Download JSON</a> &middot; <a id="rttm" download="diarization.rttm">Download RTTM</a></p>
<p><a href="/docs">API documentation</a></p>
<script>
const $ = id => document.getElementById(id);
async function run(blob, name){
  const fd = new FormData(); fd.append("file", blob, name);
  if ($("n").value) fd.append("num_speakers", $("n").value);
  const headers = $("key") ? {"X-API-Key": $("key").value} : {};
  $("go").disabled = $("sample").disabled = true; $("out").hidden = $("dl").hidden = true;
  $("status").textContent = "Processing " + name + " ...";
  try {
    const r = await fetch("/transcribe", {method: "POST", body: fd, headers});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    $("out").textContent = j.transcript; $("out").hidden = false;
    $("json").href = URL.createObjectURL(new Blob([JSON.stringify(j, null, 1)], {type: "application/json"}));
    $("rttm").href = URL.createObjectURL(new Blob([j.rttm], {type: "text/plain"}));
    $("dl").hidden = false;
    $("status").textContent = j.duration_s + " s of audio, processed in " + j.processing_s.total + " s. Roles: " +
      Object.values(j.roles).join(", ");
  } catch (e) { $("status").textContent = "Error: " + e.message; }
  $("go").disabled = $("sample").disabled = false;
}
$("f").onsubmit = e => { e.preventDefault(); const f = $("file").files[0]; if (f) run(f, f.name); };
$("sample").onclick = async () => run(await (await fetch("/samples/consult_B.wav")).blob(), "consult_B.wav");
</script></body></html>"""
