# Bengali doctor–patient transcription: diarization + ASR + roles

AIMScribe Research Associate (Speech AI) assessment · Sahil al Farib · October 2026  
Code, data preparation, results: this repository (see README). Hardware: laptop RTX 3050 4 GB (inference), Google Colab T4 16 GB (training, large models).

## 0. Evaluation protocol

| Data | Content | Used for |
|---|---|---|
| OpenSLR 53 (subset) | read Bengali, 80 min train / 18 min test, speaker-disjoint | ASR fine-tuning (Task 2) |
| Simulated conversations | 10 × 76 s from OpenSLR speakers, 2–3 speakers, overlap 1–10 % | Task 1 development |
| Real role-play consult A + B | 2.2 min, 2 speakers, my own scripts and recordings; timestamps drafted from a system output, then corrected by ear | **development** (model choice, role rules) |
| **Indic DiarBench Bengali** (public, CC BY 4.0) | real meetings / far-field / in-the-wild, 2–8 speakers, human labels, 12.6 h | **dev 3.3 h / test 9.3 h**, split by recording |
| **Synthetic clinical tests** (Indic Parler-TTS, 6 voices) | 600 sentence clips with drug/dose/test names; 6 consultations (patient first, silent patient, 2 attendants, child, nurse, overlap, fan noise) | **test**, scored once |

Rules: everything tuned on dev only; test sets scored once with frozen systems; test texts written before any system output on them (order of entries in `notes.md`).

## 1. Task 1 – Speaker diarization

**Baseline:** pyannote/speaker-diarization-3.1. **Metric:** DER (pyannote.metrics, collar 0.25 s and 0 s), plus DER inside / outside reference overlap regions (my addition).

| Data | System | DER | miss | FA | conf | DER in overlap | DER elsewhere |
|---|---|---|---|---|---|---|---|
| simulated (test half) | 3.1 | 28.1 | 2.2 | 0.0 | 25.8 | – | – |
| simulated (test half) | 3.1 + my segment re-clustering | **15.9** | 2.2 | 0.0 | **13.7** | – | – |
| DiarBench **test** 9.3 h | 3.1 | 19.8 | 6.4 | 3.1 | 10.3 | 38.2 | 18.0 |
| DiarBench **test** 9.3 h | **community-1** (chosen on dev) | **16.7** | 6.4 | 3.1 | **7.3** | **35.0** | **15.0** |

**Error analysis (real examples, `notes.md`):** wrong speaker (bn_conv_00 15.4–17.3 s), one speaker split into many (consult_B: attendant's B9/B11/B13 given a third label), speaker merge (bn_conv_01, 2 → 1), overlap (0.9 s backchannel 83 % given to the main speaker). Missed short utterances: none found in the simulated set or my real consultations — short replies are *misattributed*, not missed.

**Improvement history (what I tried and why it changed):**

1. Diagnosis on simulated data: embeddings of whole utterances separate speakers 100 %, but 3.1 embeds 10 s windows that mix 2–3 short turns → **segment-level re-clustering** (`recluster.py`): DER 28.1 → 15.9.
2. **It did not transfer to real meetings** (DiarBench dev 18.3 → 20.8–30.5): real turns are longer, and its "overlapping segments are different speakers" rule mislabels real overlap.
3. On real data the dominant error is **overlap**: DER inside overlap ≈ 2–3 × elsewhere, and missed speech never moves with any clustering setting → the second voice is not detected.
4. Overlap-aware candidates on dev: PixIT (joint diarization + separation) recovers the second voice (missed 10.2 → 3.7 %) but hallucinates on Bengali (FA 18 %); my **fusion** (community-1 + PixIT overlap, only extending existing turns) lowers overlap DER 38.0 → 33.7 but adds error just outside it (overall 16.9 → 17.6). **community-1** (better clustering) was best overall → chosen, confirmed on test (−16 % relative; speaker count right 35 → 46 of 56 clips).

Overlap in RTTM: overlapped speech is two lines covering the same time. From `data/tts_consultation/rttm/S05.rttm` (the doctor interrupts at 5.27 s while the patient speaks until 5.77 s):
```
SPEAKER S05 1 2.467 3.300 <NA> <NA> Patient <NA> <NA>
SPEAKER S05 1 5.267 1.584 <NA> <NA> Doctor <NA> <NA>
```
Reference overlap regions are also written explicitly to `data/diarbench/*/overlap/*.rttm`.

**Samples with reference and predicted annotations** (`data/samples/diarization/`, table in its README): 9 recordings, namely 4 DiarBench test clips, 2 simulated, my 2 real consultations and 1 synthetic consultation. Each has its audio, the reference RTTM, the overlap RTTM, the predicted RTTMs before and after the improvement, and a timeline plot. Examples: bengali_012, where 3.1 finds 5 speakers for 4 and community-1 finds 4 (DER 17.3 → 14.0); consult_B, where 3.1 gives the attendant a third label and community-1 does not (15.2 → 4.1); bengali_013, where the improvement does not help (14.3 → 15.9, kept on purpose).

## 2. Task 2 – Bengali ASR

**Fine-tuning:** whisper-small, full fine-tune on 73 min OpenSLR (Colab T4, 34 min, best checkpoint by dev WER).

| OpenSLR test (15 unseen speakers) | WER | CER | wrong script |
|---|---|---|---|
| whisper-small zero-shot | 144.1 | 135.7 | 98.5 % (Devanagari) |
| whisper-small fine-tuned | **66.4** | **21.4** | 0 % |

**Error analysis (real consultations):** missing end words (3 of 46 lines: A21, A25, B12; caused by the 128-token budget — Bengali costs ~2 tokens/character), short utterances (না → নাম, জি → চি, আচ্ছা → invented text), code-switching (ভাইরাল ফিভার → ওয়াই রালফিবার), medical terms (প্যারাসিটামল → পেরাসি তেমল), hallucination on silence (19/19 non-speech clips produced text), noise (CER 39 → 54 % at 10 dB).

**Improvements:** token budget 440 (no truncated lines; CER 41.6 → 39.2) and a Silero VAD gate (hallucination 19/19 → 0/19). Rejected with evidence: padding, medical prompt, VAD cropping.

**Model selection (fixed rule: lowest CER on real consultations):**

| Model | Can fine-tune | Consult CER (dev) | Clinical sentence test: CER / term accuracy |
|---|---|---|---|
| my whisper-small fine-tune | ✓ | 39.2 | – |
| tugstugi whisper-medium (+ my LoRA: no gain) | ✓ | 13.0 | 3.7 / 58 % |
| IndicConformer 120M Bengali (RNN-T / CTC) | ✓ | 9.1 / 11.1 | 4.6 / 52 % (CTC) |
| **IndicConformer 120M + my clinical fine-tune** (CTC) | ✓ | 9.4 | 3.6 / **71 %** |
| **IndicConformer 600M** | ✗ (ONNX) | **7.2** | **2.6 / 71 %** |
| Gemma 3n E2B | – | 51.4 | – |

Findings: tugstugi reproduces 88 % of OpenSLR test sentences verbatim (contamination); fine-tuning a strong model on data it has seen does not help. Ordinary Bengali is nearly solved (CER 1.5 %), **clinical terms are not (71 % of drug, dose and test names right)**.

**Clinical fine-tune of IndicConformer 120M (the trainable version; the 600M is an ONNX export).** No public Bengali clinical speech exists, so I built training data: 1,500 template sentences (drugs, doses, tests, symptoms, replies), voiced by **IndicF5 cloning 64 real OpenSLR training speakers** — a different TTS model and different voices from the test set. No test sentence and no dev (consult A/B/C) word appears in training; the test's drug and test names were split into *seen* and *never seen* (asserted in code). 1,012 clips passed a quality check (81 min), mixed 1:1 with real OpenSLR speech. CTC branch trained (the RNN-T loss's GPU kernel does not run on Colab's CUDA 13), 1,500 steps, best checkpoint by real-speech dev WER (17.8 → 15.5).

| | real consult A+B CER (dev) | medical lines | short replies | clinical test: drug/test names, seen / never seen in training |
|---|---|---|---|---|
| 120M zero-shot (CTC) | 11.1 | 12.9 | 12.9 | 50.9 / 44.1 % |
| **120M fine-tuned (CTC)** | **9.4** | **8.7** | 15.8 | **69.7 / 61.7 %** |
| 600M (pipeline) | 7.2 | 6.6 | 14.6 | 63.6 / 67.1 % |

Fine-tuning raised term accuracy by **19 points** (51.6 → 70.7 over all 570 terms; the 120 dose words such as ফাইভ হান্ড্রেড মিলিগ্রাম are in neither group, which is why the overall figure is above both), with similar gains on names it saw (+18.8) and **never saw (+17.6)**: general clinical-domain adaptation, not memorising a list. It also improved *real* consultations (CER 11.1 → 9.4). The 600M is still better on unseen names (67.1 vs 61.7). Cost: short replies got worse (few in training). By the rule fixed before training (replace the 600M only if dev CER ≤ 7.2), the pipeline keeps the 600M; the result is the evidence for the Task 3C proposal.

## 3. Task 3 – Integrated pipeline

![Figure 1. System architecture (`report/system_diagram.svg`).](system_diagram.svg)

audio → 16 kHz mono → **diarization** (community-1) → turns (same speaker, gap < 0.5 s, ≤ 25 s) → **ASR per turn** (IndicConformer, VAD gate) → **roles** (Doctor / Patient / Attendant / Other) → `Doctor [00:05–00:08]: আপনার কী সমস্যা হচ্ছে?` (+ "(overlap)" flag).

**Actual output of the final pipeline** on my real consultation B (`results/task3/final_c1/consult_B_transcript.txt`, unedited; the attendant speaks first):
```
Attendant [00:00–00:04]: স্যার আমার ছেলেকে নিয়ে এসেছি ও গতকাল থেকে পাতলা পায়খানা হচ্ছে
Doctor [00:05–00:07]: আচ্ছা বসুন বাচ্চার বয়স কত
Attendant [00:07–00:07]: চার বছর
Doctor [00:08–00:10]: দিনে কয়েকবার পাতলা পায়খানো হচ্ছে
Attendant [00:10–00:15]: প্রায় ছয় সাতবার আর রাতে একবার বমিও করেছে তারপর থেকে
Doctor [00:15–00:15]: জ্বর আছে
Doctor [00:17–00:19]: হম পর্শ ঠিক মতো হচ্ছে
Attendant [00:24–00:28]: ফার্মাসি থেকে একটা সিরাপ দিয়েছিল ম্যাট্রোনিডাজার মনে হয়
```
(The three lines between 00:16 and 00:23 are left out of the excerpt.) Every role and speaker change is right. The errors: দিনে কতবার পায়খানা → "কয়েকবার পাতলা পায়খানো" (inserted word); প্রস্রাব → "পর্শ"; and the drug name মেট্রোনিডাজল → "ম্যাট্রোনিডাজার", which is the Task 3C limitation.

**Roles** are decided from what each speaker says over the whole visit, never speaking order: polite 2nd-person verbs and clinical words → Doctor; talk about someone else (ওর, উনার, আমার ছেলে) → Attendant; the speaker's *own* symptoms (আমার/আমি within 4 words) → Patient; < 2 s of speech → Other; several attendants allowed, one patient.

| 6 synthetic consultations (test) | DER | role error | CER | cpCER | roles right* |
|---|---|---|---|---|---|
| 3.1, no improvements | 23.7 | 30.4 | 6.9 | 31.8 | 2/6 |
| 3.1 + re-clustering | 16.1 | 23.6 | 6.8 | 17.8 | 4/6 |
| community-1 (final) | 22.9 | 21.1 | 7.1 | 31.5 | 3/6 |

\* same set of roles as the truth and role error < 10 %.

Real dev consultations A + B (final pipeline, `results/task3/final_c1`): DER 2.7, role error 1.2 %, roles correct 2/2, role-attributed WER / CER 18.4 / 6.8 (cpCER 6.8). The pyannote 3.1 + re-clustering front end gives identical numbers on these two recordings. Failures understood: S03 (5 speakers) — a small cluster of nurse + child won the per-word doctor score; S06 (fan noise) — two male voices merged.

Note: re-clustering wins on synthetic speech (short clean turns, what it was built for) and loses on real speech (DiarBench dev 18.3 → 20.8–30.5, all three settings); the two tie on my real consultations. The pipeline follows the 9.3 h of real speech, not the 5 min of TTS speech. A held-out *real* consultation test (Script C) is what would settle this choice for clinics.

**Model compatibility:** all components take 16 kHz mono; diarization gives segments in seconds, ASR transcribes each turn (≤ 25 s); environments: pyannote 4 for community-1, onnxruntime for IndicConformer (README). Speed: Colab T4 ~0.9 × real time; laptop, with pyannote 4 on CPU, ~2.3 × real time (the pyannote 3.1 variant on the laptop GPU, 0.5 ×). The ONNX ASR runs on CPU in every setup.

**Major limitation – clinical vocabulary (Task 3C).** *Why:* no public Bengali clinical speech exists (checked: AmarDoctor/MedAI corpus is proprietary, DocTalkBN has YouTube-only audio with auto-captions, MediBeng is 24 sentences repeated), so even the best model gets about 3 in 10 drug, dose and test names wrong. *Component:* ASR + a post-correction step. *How:* (1) collect consented in-domain speech with a spelling guide; (2) fine-tune the trainable IndicConformer 120M on it (the synthetic version of this is done above); (3) lexicon correction against the clinic's drug list, flagged for doctor review, never silent. *Evaluate:* drug/dose accuracy on held-out real consultations, split seen/unseen drugs, plus overall CER and the false-correction rate. Second limitation: overlap (Task 1) — needs a separation model trained on Bengali.

## 4. Limitations and honesty notes

- Real consultation test is small (2.2 min, dev); synthetic tests are cleaner than real speech (optimistic); test scripts written by me.
- Role rules were fixed twice after looking at consult B (documented); validated afterwards on unseen synthetic consultations.
- Consult A/B reference timestamps were drafted from the re-clustering system's output (`prefill_labels.py`) and then corrected by ear, so DER on them (2.7) may favour the system.
- Incomplete: held-out real Script C not recorded; 60-clip TTS listening check not done (no clip was excluded by ear).
- My fine-tuned checkpoints are not in the zip because of their size; they are public on Hugging Face (`sahilfarib/aimscribe-whisper-small-bn`, `sahilfarib/aimscribe-indicconformer-120m-bn-clinical`), and the notebooks load them.

## 5. My contributions
Mine: data collection and labelling (recordings, scripts, simulated set, DiarBench eval preparation with overlap regions, synthetic test and training sets with leakage checks), the re-clustering method and its diagnosis, the overlap fusion, whisper fine-tuning recipe and fixes, error analyses, contamination detection, the fair model comparison, the pipeline and role rules, evaluation code (overlap-only DER, role-attributed WER, cpWER, term accuracy). Pretrained components used as is: pyannote, Whisper, IndicConformer, Parler-TTS, IndicF5, Silero VAD.
