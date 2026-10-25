# AIMScribe: Bengali doctor–patient transcription

Research Associate (Speech AI) technical assessment, Sahil al Farib.

A Bengali consultation recording goes in; a speaker-attributed, role-labelled transcript comes out:

```
Attendant [00:00–00:04]: স্যার আমার ছেলেকে নিয়ে এসেছি ও গতকাল থেকে পাতলা পায়খানা হচ্ছে
Doctor [00:05–00:07]: আচ্ছা বসুন বাচ্চার বয়স কত
```
(actual output on my recording `consult_B`)

| Deliverable | Where |
|---|---|
| Technical report | [`report/report.pdf`](report/report.pdf) |
| System diagram | [`report/system_diagram.png`](report/system_diagram.png) |
| Notebooks (one per task, executed, all results checked) | [`notebooks/`](notebooks/) |
| Code | `data/`, `task1_diarization/`, `task2_asr/`, `task3_pipeline/` |
| Results (metrics, RTTM, transcripts) | `results/task1`, `results/task2`, `results/task3` |
| Annotated data samples | `data/samples/`, `data/consultation/`, `data/tts*/` |
| Experiment log | [`notes.md`](notes.md) |
| Comparison with published Bengali results | [`report/sota_comparison.md`](report/sota_comparison.md) |

## Run it

Each notebook runs on a free Colab T4: it clones this repo, downloads the models and public data, re-runs the
experiments and compares every number with the submitted one. Add a Hugging Face token to Colab Secrets as
`HF_TOKEN` (and accept the model terms listed in the notebook), then Run all.

| Notebook | Runtime | Checks |
|---|---|---|
| [Task 1: diarization](https://colab.research.google.com/github/sahilaf/Aimscribe/blob/main/notebooks/task1_diarization.ipynb) | ~1.5 h | 24 / 24 ✓ |
| [Task 2: ASR](https://colab.research.google.com/github/sahilaf/Aimscribe/blob/main/notebooks/task2_asr.ipynb) | ~1 h | 27 / 27 ✓ |
| [Task 3: pipeline](https://colab.research.google.com/github/sahilaf/Aimscribe/blob/main/notebooks/task3_pipeline.ipynb) | ~30 min | 19 / 19 ✓ |

The notebooks in the repo are saved with their outputs from that run. My fine-tuned models are downloaded, not
re-trained; the training code is in the Task 2 notebook (`RUN_WHISPER_FINETUNE`, `RUN_CLINICAL_FINETUNE`).

| Hugging Face | |
|---|---|
| [sahilfarib/aimscribe-whisper-small-bn](https://huggingface.co/sahilfarib/aimscribe-whisper-small-bn) | whisper-small fine-tuned on OpenSLR 53 |
| [sahilfarib/aimscribe-indicconformer-120m-bn-clinical](https://huggingface.co/sahilfarib/aimscribe-indicconformer-120m-bn-clinical) | IndicConformer-120M, clinical fine-tune |
| [sahilfarib/aimscribe-bengali-eval](https://huggingface.co/datasets/sahilfarib/aimscribe-bengali-eval) | my recordings, synthetic test sets, samples |

## Task 1: Speaker diarization

**Data**
- *Simulated:* 10 Bengali conversations (2–3 speakers, ~76 s) built from OpenSLR 53 speakers, with overlap and
  backchannels; exact RTTM labels (`data/prepare_diarization.py`).
- *Real:* Indic DiarBench Bengali, 12.6 h, 2–8 speakers, human labels; split by recording into dev 3.3 h / test 9.3 h.
  Overlap regions are written as their own RTTM (`data/prepare_diarbench.py`).

**Model:** pyannote/speaker-diarization-3.1 (baseline). DER with pyannote.metrics, collar 0.25 s, also split into
inside / outside overlap.

**Errors found** (timelines in the notebook): wrong speaker (bn_conv_00, 15.4–17.3 s), two speakers merged
(bn_conv_01), backchannel in overlap given to the main speaker, one speaker split into three (consult_B).
No short utterance was missed outright; short replies get the wrong speaker.

**Improvements**

| Data | System | DER % | DER inside overlap % |
|---|---|---|---|
| simulated, test half | pyannote 3.1 | 28.1 | – |
| simulated, test half | + my segment re-clustering (`recluster.py`) | **15.9** | – |
| DiarBench dev (32 clips) | pyannote 3.1 | 17.8 | 40.2 |
| DiarBench dev (32 clips) | community-1 | 16.9 | 38.0 |
| DiarBench dev (32 clips) | community-1 + my PixIT overlap fusion (`overlap_fusion.py`) | 17.6 | **33.7** |
| DiarBench **test** (9.3 h) | pyannote 3.1 | 19.8 | 38.2 |
| DiarBench **test** (9.3 h) | **community-1** (chosen on dev) | **16.7** | **35.0** |

Re-clustering fixes short simulated turns but not real meetings (dev 18.3 → 24.4). The overlap fusion lowers DER
inside overlap but adds false alarms just outside it. community-1 is best overall and is used in the pipeline;
on test it also gets the speaker count right on 46 of 56 clips (3.1: 35).

**Samples:** `data/samples/diarization/`: 9 recordings, each with audio, reference RTTM, overlap RTTM, predicted RTTM
before / after, and a timeline plot.

## Task 2: Bengali ASR

**Data:** OpenSLR 53, part 0. Speaker-disjoint split: train 80 min / 65 speakers, test 18 min / 15 speakers.
16 kHz mono, silence trimmed, text normalised (`data/prepare_asr.py`; 10 samples in `data/samples/asr/`).

**Fine-tuning:** whisper-small, all weights, 800 steps, lr 1e-5, batch 16, best dev WER (Colab T4, 34 min).

| OpenSLR test | WER % | CER % |
|---|---|---|
| whisper-small zero-shot | 144.1 | 135.7 (98.5 % of outputs in the wrong script) |
| whisper-small fine-tuned | **66.4** | **21.4** |

**Errors on my real consultations:** end words lost (128-token limit), short replies (না → নাম, জি → চি),
code-switching (ভাইরাল ফিভার), medical terms (প্যারাসিটামল), hallucination on silence (19 of 19 silent clips),
noise (CER 39.2 → 44.0 / 54.4 / 61.6 at 20 / 10 / 5 dB SNR).

**Fixes (before → after, consultation CER):** token budget 128 → 440: 41.6 → 39.2; Silero VAD gate: hallucinated
clips 19 → 0.

**Model selection** (rule: lowest CER on the real consultations)

| Model | Consultation CER % | Clinical term accuracy % |
|---|---|---|
| my whisper-small fine-tune | 39.2 | – |
| tugstugi whisper-medium | 13.0 | 57.9 |
| IndicConformer 120M (CTC) | 11.1 | 51.6 |
| IndicConformer 120M + my clinical fine-tune (CTC) | 9.4 | 70.7 |
| **IndicConformer 600M** (used) | **7.2** | **71.1** |

The clinical fine-tune used 1,012 synthetic clinical sentences (IndicF5, 64 real voices) mixed with real speech;
it raised term accuracy on drug names it never saw in training from 44.1 to 61.7 %.

## Task 3: Integrated pipeline

```
audio → 16 kHz mono → community-1 diarization → speaker turns → Silero VAD + IndicConformer 600M → roles → transcript
```

**Roles** (`task3_pipeline/roles.py`) are decided from what each speaker says over the whole visit, not from who
speaks first: Doctor (polite 2nd-person, clinical words), Attendant (talks about someone else), Patient (own
symptoms), Other (under 2 s). Several attendants are allowed; overlapping turns are marked "(overlap)".

| Test | DER % | Role error % | CER % | Roles right |
|---|---|---|---|---|
| my real consultations A + B | 2.7 | 1.2 | 6.8 | 2 / 2 |
| 6 synthetic consultations, pyannote 3.1, no fixes | 23.7 | 30.4 | 6.9 | 2 / 6 |
| 6 synthetic consultations, final pipeline | 22.9 | 21.1 | 7.1 | 3 / 6 |

The synthetic consultations cover a patient speaking first, a silent patient, 5 people with 2 attendants, overlap
and fan noise. Sample transcripts: `results/task3/final_c1/*_transcript.txt`.

**Main limitation:** clinical vocabulary; about 3 in 10 drug, dose and test names are still wrong. The proposed fix
(real consented clinical speech, fine-tuning the trainable 120M model, lexicon correction flagged for review) is in
the report.

## Local setup

Tested on Windows with an RTX 3050 (4 GB) for inference; training ran on Colab.

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c constraints.txt     # pyannote 3.1, Whisper, scoring
pip install -r envs/requirements-pyannote4.txt         # separate venv: community-1 (final pipeline)
```
`envs/requirements-nemo.txt` (IndicConformer-120M training) and `envs/requirements-tts.txt` (test-audio generation)
are Colab-only environments. Put a Hugging Face token in `.env` as `HF_TOKEN=...`; `python check_setup.py` checks
the GPU, libraries and gated-model access. Audio and models live outside the repo in `~/aimscribe_data`; the ASR
model goes to `~/aimscribe_data/models/indic-conformer-600m`:
```bash
python -c "import os; from huggingface_hub import snapshot_download as d; d('ai4bharat/indic-conformer-600m-multilingual', local_dir=os.path.expanduser('~/aimscribe_data/models/indic-conformer-600m'))"
```

Run the pipeline on any recording (in the pyannote 4 environment):
```bash
python task3_pipeline/pipeline.py --audio my_recording.m4a --out_dir results/task3/my_run
```
It writes `<name>_transcript.txt`, `<name>.rttm` and `<name>.json` (turns, overlap flags, role evidence).
`--num_speakers N` sets the speaker count; `--baseline` turns all improvements off (main environment).

## Repository

| Folder | Contents |
|---|---|
| `data/` | dataset preparation, synthetic data generators, reference labels, Bengali text normalisation |
| `task1_diarization/` | diarization runs, DER (inside / outside overlap), error analysis, re-clustering, overlap fusion |
| `task2_asr/` | ASR wrappers, fine-tuning (Whisper, IndicConformer-120M), WER / CER, error analyses, term accuracy |
| `task3_pipeline/` | pipeline, role rules and tests, pipeline evaluation (DER, role error, cpWER) |
| `notebooks/` | one Colab notebook per task, plus the one that generated the synthetic test audio |
| `results/` | every run quoted in the report, one folder per run |
| `report/` | report, diagram, comparison with published results |

## Contributions

My work: the consultation recordings and their labels, the evaluation data and dev / test protocol, segment
re-clustering, overlap fusion, the fine-tuning recipes and synthetic clinical training data, the error analyses,
the model comparison, the pipeline and the role rules. Pretrained models used as they are: pyannote, Whisper,
IndicConformer, Parler-TTS, IndicF5, Silero VAD. Public data: OpenSLR 53 (CC BY-SA 4.0), Indic DiarBench (CC BY 4.0).
