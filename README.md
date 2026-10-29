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

## Live demo

The final pipeline is deployed on Modal (T4 GPU, starts on demand):
**https://sahilaf--aimscribe-aimscribe-web.us-east.modal.run**. Upload a recording on the page, or call the API:

```bash
curl -X POST https://sahilaf--aimscribe-aimscribe-web.us-east.modal.run/transcribe \
     -H "X-API-Key: <access key>" -F "file=@consultation.m4a" -F "num_speakers=2"   # num_speakers optional
```
The response is JSON: `transcript`, `turns` (speaker, role, start, end, overlap, text), `roles`, `duration_s`,
`processing_s`, `rttm`. Interactive documentation: `/docs`. The access key was sent with the submission. After an idle
period the first request takes about a minute to start; a one-minute recording then takes about 6 s.
Deployment code: `deploy/modal_app.py` (`modal deploy deploy/modal_app.py`).

## Run it

Each notebook runs on a free Colab T4: it clones this repo, downloads the models and public data, re-runs the
experiments and compares every number with the submitted one. Add a Hugging Face token to Colab Secrets as
`HF_TOKEN` (and accept the model terms listed in the notebook), then Run all.

| Notebook | Runtime | Checks |
|---|---|---|
| [Task 1: diarization](https://colab.research.google.com/github/sahilaf/Aimscribe/blob/main/notebooks/task1_diarization.ipynb) | ~1.5 h | 24 / 24 ✓ |
| [Task 2: ASR](https://colab.research.google.com/github/sahilaf/Aimscribe/blob/main/notebooks/task2_asr.ipynb) | ~1 h | 27 / 27 ✓ |
| [Task 3: pipeline](https://colab.research.google.com/github/sahilaf/Aimscribe/blob/main/notebooks/task3_pipeline.ipynb) | ~30 min | 19 / 19 ✓ |

The checks confirm the reported numbers can be regenerated, not that they generalise. The default run evaluates the
saved fine-tuned models and the saved PixIT output rather than re-training or re-running them.

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
| DiarBench dev (32 clips) | community-1 + my PixIT overlap fusion (`overlap_fusion.py`), not used | 17.6 | **33.7** |
| DiarBench **test** (9.3 h) | pyannote 3.1 | 19.8 | 38.2 |
| DiarBench **test** (9.3 h) | **community-1** (chosen on dev) | **16.7** | **35.0** |

Re-clustering fixes short simulated turns but not real meetings (dev 18.3 → 24.4). The overlap fusion is an
overlap-only gain: it lowers DER inside overlap but raises overall DER, so the final system does not use it.
community-1 (without fusion) is best overall and is used in the pipeline; on test it also gets the speaker count
right on 46 of 56 clips (3.1: 35).

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

**Model selection** (rule: lowest CER on the real consultations). The three test sets (OpenSLR read speech, my 2.2 min
of real consultations, the synthetic TTS clinical test) are different data; their numbers are not comparable with each other.

| Model | Real consultations, CER % | Synthetic clinical test, term accuracy % |
|---|---|---|
| my whisper-small fine-tune | 39.2 | – |
| tugstugi whisper-medium | 13.0 | 57.9 |
| IndicConformer 120M (CTC) | 11.1 | 51.6 |
| IndicConformer 120M + my clinical fine-tune (CTC) | 9.4 | 70.7 |
| **IndicConformer 600M** (used) | **7.2** | **71.1** |

Term accuracy = share of the 570 drug, dose and test-name occurrences in the synthetic test that appear whole in the
output (no partial credit). The clinical fine-tune used 1,012 synthetic clinical sentences (IndicF5, 64 real voices)
mixed with real speech; on the synthetic test it raised accuracy on drug names never seen in training from 44.1 to
61.7 %. This is synthetic-speech evidence, not clinical validation.

## Task 3: Integrated pipeline

```
audio → 16 kHz mono → community-1 diarization → speaker turns → Silero VAD + IndicConformer 600M → roles → transcript
```

**Roles** (`task3_pipeline/roles.py`) are decided from what each speaker says over the whole visit, not from who
speaks first: Doctor (polite 2nd-person, clinical words), Attendant (talks about someone else), Patient (own
symptoms), Other (under 2 s). Several attendants are allowed; overlapping turns are marked "(overlap)".

| Test | DER % | Role error % | CER % | Roles right |
|---|---|---|---|---|
| my real consultations A + B (development, not an independent test) | 2.7 | 1.2 | 6.8 | 2 / 2 |
| 6 synthetic consultations, pyannote 3.1, no fixes | 23.7 | 30.4 | 6.9 | 2 / 6 |
| 6 synthetic consultations, pyannote 3.1 + re-clustering | 16.1 | 23.6 | 6.8 | 4 / 6 |
| 6 synthetic consultations, final pipeline (community-1) | 22.9 | 21.1 | 7.1 | 3 / 6 |

The synthetic consultations cover a patient speaking first, a silent patient, 5 people with 2 attendants, overlap
and fan noise. Re-clustering does better on these short, clean synthetic turns; community-1 is better on 9.3 h of real
meetings; on real consultations the choice is not yet settled (no held-out real recording). The A + B references were
drafted from system output and corrected by ear. Sample transcripts: `results/task3/final_c1/*_transcript.txt`.

Applied to the reference speakers and text, the role rules are right in 8 of 8 consultations. Every end-to-end failure
is a room with 3 or more people where diarization merged two speakers (S03, S04, S06): the weak point is speaker
counting, not the role rules.

**Main limitation:** clinical vocabulary; on the synthetic test about 3 in 10 drug, dose and test names are still wrong. The proposed fix
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

## Reproduce from the command line

The same steps the notebooks run. `D=~/aimscribe_data`; every run writes `results/task*/<run>/`.
`(p4)` = pyannote 4 environment, `(NeMo)` = `envs/requirements-nemo.txt` (Colab).

**Task 1**
```bash
python data/prepare_diarization.py --n_clips 10                    # simulated conversations + RTTM
python task1_diarization/diarize.py --run baseline                 # pyannote 3.1
python task1_diarization/evaluate.py --run baseline                # DER
python task1_diarization/analyze_errors.py --run baseline          # error timelines
python task1_diarization/recluster.py --src baseline --run rc_t0.70 --threshold 0.70
python task1_diarization/evaluate.py --run rc_t0.70
python task1_diarization/compare.py baseline rc_t0.70              # before / after
python data/prepare_diarbench.py --dev_fraction 0.4                # DiarBench dev / test split + overlap RTTM
python task1_diarization/diarize.py --run db_test_baseline --wav_dir $D/diarbench/test --ref_dir data/diarbench/test/rttm
python task1_diarization/diarize_overlap_aware.py --system community1 --run db_test_community1 --wav_dir $D/diarbench/test   # (p4)
python task1_diarization/evaluate.py --run db_test_community1 --ref_dir data/diarbench/test/rttm --overlap_dir data/diarbench/test/overlap
python task1_diarization/diarize_overlap_aware.py --system pixit --run db_dev_pixit --wav_dir $D/diarbench/dev --chunk_s 300
python task1_diarization/overlap_fusion.py --base db_dev_community1 --pixit db_dev_pixit --run db_dev_fuse_mo0.0_adj0.0 --adjacent_s 0.0
python data/make_diarization_samples.py                            # data/samples/diarization/
```

**Task 2** (fine-tuned model: `python -c "from huggingface_hub import snapshot_download as d; d('sahilfarib/aimscribe-whisper-small-bn', local_dir='FT')"`)
```bash
python data/prepare_asr.py --train_minutes 90 --test_minutes 20
python task2_asr/evaluate_asr.py --model openai/whisper-small --run baseline_small
python task2_asr/finetune_whisper.py --run ft_small --max_steps 800 --lr 1e-5 --batch 8 --grad_accum 2   # Colab T4, ~35 min
python task2_asr/evaluate_asr.py --model FT --run ft_small_test
python task2_asr/analyze_asr_errors.py --run ft_small_test
python task2_asr/eval_consult.py --model FT --run ft_consult                                    # before
python task2_asr/eval_consult.py --model FT --max_new_tokens 440 --vad_gate --run ft_consult_tok440_vadgate   # after
python task2_asr/eval_consult.py --model FT --max_new_tokens 440 --noise_snr 10 --run ft_consult_tok440_noise10db
python task2_asr/hallucination_probe.py --model FT
python task2_asr/eval_consult.py --backend indicconformer --model $D/models/indic-conformer-600m --run zs_indicconformer600m_consult
mkdir -p $D/tts_sentences && cp data/tts/audio/*.wav $D/tts_sentences/   # synthetic clinical test
python task2_asr/eval_tts_sentences.py --backend indicconformer --model $D/models/indic-conformer-600m --run tts_sent_indicconformer
python task2_asr/finetune_indicconformer.py --max_steps 1500 --val_every 250 --batch 8 --lr 5e-5 --run icft_120m   # (NeMo)
```

**Task 3**
```bash
python task3_pipeline/test_roles.py                                # role-rule cases
python task3_pipeline/pipeline.py --audio data/consultation/audio/consult_A.wav data/consultation/audio/consult_B.wav --out_dir results/task3/final_c1   # (p4)
python task3_pipeline/evaluate_pipeline.py --run final_c1
python task3_pipeline/pipeline.py --audio data/tts_consultation/audio/S0*.wav --out_dir results/task3/tts_final_c1   # (p4)
python task3_pipeline/evaluate_pipeline.py --run tts_final_c1 --set tts_consultation
```

## Repository

| Folder | Contents |
|---|---|
| `data/` | dataset preparation, synthetic data generators, reference labels, Bengali text normalisation |
| `task1_diarization/` | diarization runs, DER (inside / outside overlap), error analysis, re-clustering, overlap fusion |
| `task2_asr/` | ASR wrappers, fine-tuning (Whisper, IndicConformer-120M), WER / CER, error analyses, term accuracy |
| `task3_pipeline/` | pipeline, role rules and tests, pipeline evaluation (DER, role error, cpWER) |
| `notebooks/` | one Colab notebook per task, plus the one that generated the synthetic test audio |
| `results/` | every run quoted in the report, one folder per run (parameter sweeps and superseded runs are in the GitHub repository) |
| `report/` | technical report and system diagram |

## Contributions

My work: the consultation recordings and their labels, the evaluation data and dev / test protocol, segment
re-clustering, overlap fusion, the fine-tuning recipes and synthetic clinical training data, the error analyses,
the model comparison, the pipeline and the role rules. Pretrained models used as they are: pyannote, Whisper,
IndicConformer, Parler-TTS, IndicF5, Silero VAD. Public data: OpenSLR 53 (CC BY-SA 4.0), Indic DiarBench (CC BY 4.0).
