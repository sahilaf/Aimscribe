# Experiment log

Every run, number, problem and decision, in the order they happened.

## Phase 0: Setup (2026-10-20)
- Local machine: Windows 11, Python 3.12, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), ffmpeg installed
  - Plan: run diarization + ASR inference locally; 4 GB is too small to fine-tune Whisper, so do that on Colab
- Colab: Python 3.13, 13.6 GB RAM, free T4 GPU (16 GB) for fine-tuning
- HF token stored in `.env` locally (git-ignored) and in Colab Secrets; never written in code
- Gated models: pyannote/segmentation-3.0 OK; pyannote/speaker-diarization-3.1 needs its own terms accepted

## Decisions
| Decision | Choice | Why |
|---|---|---|
| Diarization model | pyannote/speaker-diarization-3.1 | pretrained, handles overlap, easy DER eval with pyannote.metrics |
| ASR model | openai/whisper-small | multilingual (supports Bengali), fits a T4 for fine-tuning |

## Phase 1: Data (2026-10-20)
- Source: **OpenSLR 53** (Large Bengali ASR corpus, read speech), part 0 only (13,758 utterances). License CC BY-SA 4.0.
- OpenSLR downloads at ~15 KB/s from my laptop vs ~7.5 MB/s on Colab, so data prep runs on Colab.
- Raw/processed audio kept outside the repo (`~/aimscribe_data` locally, `/content/aimscribe_data` on Colab).

### ASR set (`data/asr/`)
| split | utts | speakers | minutes |
|---|---|---|---|
| train | 1623 | 65 | 79.9 |
| test | 390 | 15 | 18.2 |
- Speaker-disjoint split (overlap = 0), so test WER measures new voices.
- Dropped 645 utterances whose text has digits/Latin letters (ambiguous to read).
- Preprocessing: mono 16 kHz, trim silence (top_db=35) keeping 0.15 s pad, RMS-normalise to -20 dBFS.
- Text: NFC normalise, strip punctuation incl. danda, remove ZWJ/ZWNJ.

### Diarization set (`data/diarization/`)
- No public Bengali conversational corpus with speaker labels, so I simulated 10 conversations from
  OpenSLR speakers NOT used in ASR training (exact reference RTTM by construction).
- 6 x 2-speaker + 4 x 3-speaker, ~76 s each (~12.7 min total); overlap 1-10% of speech;
  short backchannel segments; per-speaker gain -4..0 dB; last 2 clips with white noise at 15 dB SNR.
- Limitation to report: read speech stitched together isn't spontaneous conversation (no
  natural turn-taking, the same recording channel per speaker); real-world DER will be higher.

### Consultation recordings (2026-10-20)
- Recorded by 2 people on a phone (AAC, 48 kHz stereo) and converted to 16 kHz mono WAV.
- `consult_A` = Script A (fever, doctor speaks first), 76 s. `consult_B` = Script B (child with diarrhoea, attendant speaks first, patient silent), 57 s.
- Followed the script; no line changes.
- Both have the same background level (noise floor about -57 dBFS), so they're clean. No noisy take yet.

### Early observation for Task 2 (baseline Whisper-small, NOT fine-tuned)
- On the first 12 s of both recordings, whisper-small with `language="bn"`:
  - fp16 pipeline: degenerate repetition ("ರಿರಿರಿ…", "ḍḍḍ…"), a classic hallucination loop
  - fp32 with beam=5 and no_repeat_ngram_size=3: output in **Kannada script** instead of Bengali
- Shows why fine-tuning is needed; use it as a real error example (wrong script / hallucination).

### Reference-label bug found while validating Phase 2 (2026-10-20)
- First pyannote smoke test on bn_conv_00: DER 58%, of which **missed speech 49%**, which is suspicious for clean audio.
- Checks: (1) passing a file path vs audio in memory gave the same result, so it isn't my input code;
  (2) pyannote's speech detector (segmentation-3.0) alone found 51%; (3) a completely different VAD (Silero) found only 43% of speaker 2b883's labelled speech.
- Cause: the reference labelled each whole energy-trimmed utterance (librosa top_db=30) as speech. OpenSLR phone recordings have a noise floor
  only ~20 dB below the voice, so noisy lead-ins, tails and pauses were labelled "speech".
- Fix (v2 of prepare_diarization.py): Silero VAD on each clean single-speaker utterance before mixing; pauses < 0.3 s merged (annotator convention);
  utterance audio cut to first..last speech. Transcripts JSON keeps whole utterances (for Task 3), RTTM keeps speech-only regions.
- Lesson: validate the references first; two independent detectors agreeing with each other against my labels showed the labels were wrong.

### Problems hit
- `list.remove()` on dicts holding numpy arrays raised a ValueError, so backchannels are now removed by index.
- pip upgraded torch to a CPU build while installing pyannote deps, fixed with `constraints.txt`.

## Phase 2: Task 1 baseline (2026-10-20)
- Model: pyannote/speaker-diarization-3.1, default params (clustering threshold 0.7046, min_cluster_size 12, min_duration_off 0.0).
- Hardware: RTX 3050 laptop GPU, about 3.5 s per 76 s clip (real-time factor about 0.05).
- Data: 10 simulated clips (v2 reference labels, speech-only regions from Silero VAD).

| collar | DER | missed | false alarm | confusion |
|---|---|---|---|---|
| 0.25 s | **25.55%** | 3.11% | 0.11% | **22.33%** |
| 0.0 s | 31.31% | 8.57% | 1.66% | 21.09% |

Observations:
- **Confusion dominates** (22 of the 25.6 points): speech is found, but given to the wrong speaker.
- **Wrong speaker count in 6 of 10 clips**: bn_conv_01 merged 2 speakers into 1 (DER 48%); bn_conv_02 split 2 into 3; all four
  3-speaker clips except bn_conv_06 got 2 speakers (07, 08, 09).
- Even with the right count, clips 00, 05 and 06 have 20-25% confusion, so utterances are being attached to the wrong cluster.
- Hypothesis: utterances are short (median ~1.6 s; 11-26 per clip are under 1.6 s), so speaker embeddings are noisy, and the
  speakers are all similar phone recordings. That makes the clustering step the weak point, which points the improvement there.
- The collar matters for missed speech (3.1% vs 8.6%): most boundary errors are within 0.25 s.

### Error analysis, baseline (`task1_diarization/analyze_errors.py`, results/task1/baseline/utterance_errors.csv + plots/)
358 reference utterances: 260 ok, 48 speaker_merge, 39 wrong_speaker, 6 speaker_split, 4 overlap, 1 missed.
Concrete examples (timestamps in the clip):
1. **Speaker merge (one cluster for two people)**: bn_conv_01, a 2-speaker clip predicted as 1 speaker, so every turn of c4973 is confusion; DER 48% (plots/bn_conv_01.png).
2. **One speaker as multiple**: bn_conv_02, 41.60-43.80 s, speaker 16877 "সরি বলতে থাকবে" assigned to an extra cluster SPEAKER_02 (3 predicted vs 2 real).
3. **Incorrect speaker assignment**: bn_conv_00, 15.38-17.28 s, 2b883 "শিল্পকেন্দ্র হিসেবে কাজ করে" fully given to 61820, even though the speaker count is right.
4. **Overlap**: bn_conv_00, 18.26-19.16 s, a 0.9 s backchannel "আব্দুস সবুর" spoken over 61820's turn; 83% given to 61820.
5. **Missed short utterances**: almost none in the simulated data. Short utterances are *misattributed*, not missed (baseline: 32% of short
   vs 22% of long utterances handled badly; 5/7 overlapped backchannels were detected correctly). Check on the real recordings.

### Experiments (tuned on dev = even clips, reported on test = odd clips; collar 0.25)
| run | what changed | dev DER | test DER | all DER | right #spk |
|---|---|---|---|---|---|
| baseline | pyannote 3.1 defaults | 23.01 | 28.08 | 25.55 | 5/10 |
| oracle_spk | true speaker count given to pyannote | 19.54 | 26.93 | 23.25 | 9/10 |
| thr_0.65 / 0.75 | pyannote clustering threshold | 19.65 / 19.65 | 32.72 / 27.49 | 26.20 / 23.58 | 5/10 |
| mcs_2 / mcs_6 | min_cluster_size | 24.43 / 23.63 | 28.96 / 27.67 | 26.70 / 25.66 | 2/10, 5/10 |
| **rc_t0.70** | **segment-level re-clustering (mine), threshold from dev** | **10.63** | **15.93** | **13.29** | 7/10 |
| **rc_oracle** | re-clustering + true speaker count | **6.14** | **7.35** | **6.75** | 10/10 |
Strict collar 0: baseline 31.31 → rc_t0.70 19.78 → rc_oracle 13.91 (confusion 21.1 → 9.5 → 3.7).
- Tuning pyannote's own threshold is unstable (dev-best 0.65 is worse on test), so it isn't a real fix.
- Telling pyannote the true count fixes count errors (bn_conv_02 23.8 → 4.2) but leaves 20-42% confusion on clips 00/01/05.

### Diagnosis that led to the improvement
- wespeaker embeddings computed on WHOLE clean reference utterances (>= 1 s, no overlap): 1-nearest-neighbour speaker accuracy is
  **100% on all 10 clips**, mean cosine within-speaker ~0.55 vs between ~0.12. So the embeddings CAN separate these speakers.
- pyannote 3.1 embeds fixed 10 s windows; with 1-3 s turns each window holds 2-3 speakers and the local-to-global
  speaker assignment gets mixed up. Hence: keep pyannote's segmentation, re-label with one embedding per segment
  (`task1_diarization/recluster.py`), cluster reliable (>= 1 s) segments with average-linkage cosine, attach short segments to the nearest centroid,
  and forbid two overlapping segments from sharing a speaker.
- After the fix, **overlap is the main remaining error**: rc_oracle handles 21% of overlap-touching utterances badly vs 1% of the rest.
  That's the limitation to carry into Task 3.
- Caveat for the report: simulated read speech; real conversations have longer turns (which help pyannote's windows) and
  more channel variation. Validate on the real consultation recordings.

## Real consultation recordings: reference + diarization (2026-10-21)
- Draft labels made automatically (diarization + Needleman-Wunsch alignment to the script), then **checked by ear in Audacity** by me.
  IDs were restored to the script numbering
  (A17 was split into A17a "আচ্ছা" + A17b after a ~2.5 s pause).
- What actually happened vs the script: the scripted overlaps A9/A8 and A24/A23 did NOT happen in the recording (only A10 overlaps A9 by ~0.2 s);
  B15 "আচ্ছা" came AFTER B14 (44.09-45.04 s), not over it. Diarization merged it into the attendant's next turn (B16),
  so it's the same speaker and costs no DER. **It is not a missed short utterance.** No missed short utterances on real audio either.

| run (collar 0.25) | consult_A | consult_B | overall | #spk predicted (A, B) |
|---|---|---|---|---|
| pyannote default (no count) | 1.61 | **15.23** | 7.43 | 2, **3** |
| pyannote + num_speakers=2 | 1.61 | 4.07 | 2.66 | 2, 2 |
| **re-clustering, threshold 0.70 (no count)** | 1.61 | **4.07** | **2.66** | 2, 2 |

- **Real-audio "one speaker as multiple" example:** in consult_B, default pyannote gave the attendant's lines B9 "না, কম হচ্ছে" (20.0-20.9 s),
  B11 "ফার্মেসি থেকে…মেট্রোনিডাজল" (24.6-28.9 s) and B13 "জি" (38.3 s) to an extra speaker SPEAKER_00. Re-clustering merged them back without being told the count.
- Real turns are longer and cleaner than the simulated ones, so DER is much lower overall (consult_A is near-perfect even for the baseline).
- **Caveat for the report:** the reference started from the system's own boundaries, and most were accepted unchanged after listening,
  so miss/false-alarm numbers on these 2 recordings are likely optimistic (anchoring bias). The **speaker-assignment** comparison
  (confusion 11.2% → 0%) is unaffected, because labels were checked against the script roles. Only 2 short recordings: illustrative, not statistically strong.

## Phase 3: Task 2 Bengali ASR (2026-10-21)
Environment (Colab): Python 3.13, torch 2.11 + CUDA 13, transformers 5.20, Tesla T4 15 GB. Laptop: transformers 4.46 (scripts work with both).

### Baseline: zero-shot openai/whisper-small, test set (390 clips, 15 unseen speakers, 18 min)
| WER | CER | wrong script | empty | repetition loops | over-long output |
|---|---|---|---|---|---|
| 144.1% | 135.7% | **98.5%** | 1.5% | 4.1% | 14.6% |
- Output is phonetically right but written in **Devanagari (Hindi script)**: ref "নৌকায় না সাঁকো বেঁধে" → "नुखाय ना शाकोवेदे".
  On the role-play recordings, earlier tests gave Kannada script.
- Checked it's NOT a bug: same output with language="bn", "bengali" or none, and with the decoder prompt forced by hand to
  `<|startoftranscript|><|bn|><|transcribe|><|notimestamps|>` (ids 50258 50302 50359 50363). whisper-small simply has too little
  Bengali in pre-training to keep the right script, so WER/CER > 100% (every word wrong, plus insertions).

### Fine-tuning (task2_asr/finetune_whisper.py)
- whisper-small, full fine-tune, lr 1e-5, warmup 50, 800 steps, batch 8 x grad-accum 2 (batch 16 ran out of memory on the T4), fp16.
- Train 1,4xx utts from 59 speakers / dev = 6 held-out TRAIN speakers (speaker-disjoint); best checkpoint chosen by dev WER.
- Time: 34 min on the T4 (1.76 s/step).

| step | train loss | dev WER | dev CER |
|---|---|---|---|
| 100 | 0.904 | 79.15 | 28.42 |
| 200 | 0.456 | 67.49 | 22.51 |
| 300 | 0.207 | 64.57 | 19.56 |
| 400 | 0.093 | 57.62 | 17.53 |
| 500 | 0.047 | 60.31 | 19.75 |
| **600** | 0.027 | **56.95** | **17.30** |
| 700 | 0.008 | 58.30 | 17.81 |
| 800 | 0.004 | 57.85 | 18.23 |
- Overfitting: train loss → 0.004 while dev WER plateaus from step ~400. With 73 min of data the model memorises the training sentences.
  More data, SpecAugment or fewer steps would help; the checkpoint was chosen by dev WER.
- "missing key proj_out.weight" warning at reload: harmless (tied to the decoder embeddings); confirmed by sensible test results.

### Test set: baseline vs fine-tuned (same script, greedy decoding)
| model | WER | CER | wrong script | repetition loops | over-long |
|---|---|---|---|---|---|
| whisper-small zero-shot | 144.1 | 135.7 | 98.5% | 4.1% | 14.6% |
| **whisper-small fine-tuned** | **66.4** | **21.4** | **0.0%** | 0.0% | 1.0% |
- CER 21% vs WER 66%: most words are *nearly* right (a wrong suffix or vowel sign fails the whole word). Bengali morphology makes WER harsh.
- Test WER (66.4) > dev WER (57.0): test speakers differ more / harder recordings. Dev was used for checkpoint choice, so test is the honest number.

### Error analysis: test set (fine-tuned, `analyze_asr_errors.py`, results/task2/ft_small_test/error_analysis.json)
- First word deleted in 0.0% of clips, **last word deleted 9.0%**, middle word 6.5%, so the ends of segments lose words more often.
- 5.9 inserted words per 100 reference words.
- WER by length: 1-2 words 74.2% | 3 words 67.9% | 4 words 59.2% | 5+ words 68.5% (very short clips are the worst: no context).
- Typical errors are near-misses in spelling: পুরস্কার→পুরশকার, সমুদ্রে→সমুদ্ধে, কেন→কেনো; word-splitting: একইসঙ্গে→একই সঙ্গে, তথ্যপ্রমাণ→তথ্য প্রমানু
  (each split costs 2 word errors, which explains WER 66% vs CER 21%).

### Error analysis: real role-play consultations (line by line, `eval_consult.py`, cut at reference timestamps, greedy)
| model | ALL WER/CER | SHORT | EN (code-switch) | MED | OVERLAP |
|---|---|---|---|---|---|
| zero-shot whisper-small | 179.8 / 132.3 | 229.6 / 177.5 | 189.2 / 127.8 | 192.9 / 125.9 | 100.0 / 97.1 |
| fine-tuned (max_new_tokens 128) | 82.0 / 41.6 | 85.2 / 49.2 | 73.5 / 37.2 | 81.6 / 39.5 | 83.3 / 35.3 |
- Real recordings are much harder than the test set (CER 41.6 vs 21.4): conversational speech, a new microphone and voices, and vocabulary unlike news-style read speech.
Concrete error examples (all categories from the PDF):
1. **Short utterances**: A10 "না"→"নাম" (extra letter); A20 "আচ্ছা"→"আদ সাম্পিক" (invented); B13 "জি"→"চি";
   sentence-initial "হুম" dropped in A5, A16, B8, i.e. **missing word at the beginning**. (A14 "না" and A24 "জি" were correct.)
2. **Medical terms**: অ্যামলোডিপিন ফাইভ মিলিগ্রাম→"এম্ডাড়িপিন ফাইর নিলেগ্রাঙ"; প্যারাসিটামল→"পেরাসি তেমল"; মেট্রোনিডাজল→"ম্যাক্ঠনি ডাজয়";
   অ্যান্টিবায়োটিক→"অন্টিভিটিক"/"এনটিবেয়ারটিক".
3. **Code-switching**: ভাইরাল ফিভার→"ওয়াই রালফিবার"; সিবিসি→"চিভিসি"; হাই ব্লাড প্রেশার→"হাই ভালাক প্রশও"; টেস্ট→"টেষ্ট" (spelling variant).
4. **Missing words at the end (decoding bug)**: A21, A25 and B12 output ended in a broken character "�" and lost their last words.
   Cause: Whisper's byte-level BPE spends **~1.97 tokens per Bengali character**, so lines over ~64 characters exceed max_new_tokens=128.
5. **Hallucination**: with NO speech, zero-shot outputs "अप अप अप…" (100+ times) and fine-tuned invents Bengali-looking nonsense
   (3 s digital silence → "যেয়ায়" x15; every real pause between lines → "ক্যাসায়াসায়…"). 19/19 non-speech clips produced text.
6. **Background noise** (white noise added to the consultation lines, fine-tuned): CER clean 39.2 → 20 dB 43.0 → 10 dB 54.1 → 5 dB 61.0.

### Improvements (one change at a time)
| change | consult WER / CER | test WER / CER | notes |
|---|---|---|---|
| fine-tuned, max_new_tokens 128 (before) | 82.0 / 41.6 | 66.36 / 21.44 | |
| **+ token budget 440** | **80.9 / 39.2** | **66.20 / 20.67** | EN CER 37.2→31.3, MED 39.5→35.0; no more truncated "�" lines |
| + 0.2 s padding around cuts | 83.5 / 38.8 | – | short replies better (CER 49→45), but pulls in the other speaker's words. Not adopted |
| + medical prompt (153 tokens) | 80.9 / 39.7 | – | MED CER 35.0→34.3 only; OVERLAP worse; prompt words came from the script (optimistic). **Rejected** |
| + beam search 5 | 79.8 / 38.0 | 64.63 / 20.00 | small consistent gain, 4.4x slower |
| **+ VAD gate (Silero, gate only)** | 80.9 / 39.2 (unchanged) | – | **hallucination on non-speech: 19/19 clips, 49 words → 0/19, 0 words** |
| VAD gate + crop to speech | 81.2 / 39.9 | – | crop cut real words (B16 lost "স্যার স্টুল টেস্ট"; A20 "আচ্ছা" dropped at 200 ms min speech). **Rejected** |
- Final ASR config for Task 3 (`task2_asr/asr.py`): fine-tuned whisper-small, max_new_tokens 440, VAD gate (min speech 100 ms, no crop), greedy
  (beam 5 optional where latency allows).
- Not fixed (future work): medical vocabulary and code-switching need domain data (doctor–patient speech, English medical terms
  written in Bengali script); noise robustness needs noise augmentation in training.
- Model transfer: v5-saved processor_config.json confused transformers 4.46 on the laptop, so it was renamed to processor_config.v5.json and
  standard preprocessor/tokenizer files were saved (the tokenizer/feature extractor are unchanged by fine-tuning).

## Phase 4: Task 3 integrated pipeline (2026-10-21)
Code: `task3_pipeline/pipeline.py` (end to end), `roles.py`, `evaluate_pipeline.py` (real recordings), `evaluate_sim.py` (simulated clips),
`test_roles.py` (role edge cases). Diagram: `report/system_diagram.svg`. Transcripts: `results/task3/final/consult_*_transcript.txt`.

### Design: diarize first, then transcribe each turn
- Order: audio → pyannote segmentation → segment-level re-clustering → turns (same speaker, gap < 0.5 s, max 25 s) → ASR per turn
  (fine-tuned whisper-small, VAD gate, 440 tokens) → roles → transcript `Role [mm:ss–mm:ss]: text` (+ "(overlap)" flag).
- Why this order: each ASR input holds one speaker and stays under Whisper's 30 s window; timestamps come from diarization.
  Cost: a wrong diarization cut becomes an ASR error, and overlapped speech is transcribed from mixed audio.
- Compatibility: both models take 16 kHz mono (one ffmpeg resample); pyannote gives segments in seconds, Whisper takes ≤30 s
  clips; everything runs in one Python env (torch 2.5.1 + CUDA, pyannote 3.3.2, transformers 4.46). Laptop RTX 3050 4 GB holds both
  (pyannote + whisper-small fp16). Speed: ~0.25x real time (76 s audio in ~17-19 s).
- Model choice: pyannote 3.1 (pretrained, overlap-aware, standard DER tooling) over NeMo (heavier setup, manifest files, more GPU);
  whisper-small (multilingual incl. Bengali, fine-tunes on a free T4) over wav2vec2/Conformer (no Bengali tokenizer/LM out of the
  box, needs a CTC vocabulary built from scratch) and over whisper-medium/large (don't fit T4 fine-tuning comfortably).

### Role assignment (`roles.py`): transparent rules over the WHOLE conversation, never speaking order
- Doctor = highest score of (clinical/instruction words + polite 2nd-person verb endings − 2 x addressing-the-doctor words) per word.
- Others: Attendant if "about someone else" cues (ওর, উনার, আমার ছেলে/আব্বা, বাচ্চা) outweigh first-person symptom cues, else Patient;
  only one Patient (extra → Attendant); < 2 s of speech → Other. Fuzzy matching (ratio ≥ 0.75) because ASR CER is ~40%.
- Edge cases (test_roles.py): patient speaks first ✓, two attendants + patient + doctor ✓, silent patient + two attendants ✓,
  one-word speaker → Other ✓, noisy ASR spelling ✓ (5/5). Known weakness (fails as expected): a patient who mentions family
  ("আমার ছেলে বলল…") + an anxious attendant talking in first person → roles swapped. Keyword rules count words, not meaning.
  Caveat: tests were written by me knowing the rules; real validation needs unseen consultations.
- Overlap: overlapped turns are kept for both speakers and flagged "(overlap)" in the transcript.

### Results: real role-play recordings (hand-checked references)
| whole pipeline | DER | role error | CER all text | CER role-attributed | WER role-attributed | speakers in B |
|---|---|---|---|---|---|---|
| before (plain pyannote, 128 tokens, no VAD gate) | 7.43 | 1.18 | 43.2 | 44.5 | 83.1 | 3 ("Attendant 1/2") |
| **after (all improvements)** | **2.66** | **1.18** | **38.4** | **38.4** | **80.9** | **2** |
- Roles correct on both recordings, including consult_B (attendant speaks first, silent patient, no fake Patient created).
- Role rules applied to the REFERENCE text also give correct roles, so the rules aren't the bottleneck here.
- Role-attributed CER = plain CER: no text ends up under the wrong person; the remaining error is recognition.

### Results: 10 simulated conversations (exact per-speaker text), cpWER / cpCER (CHiME-6 style, best speaker permutation)
| whole pipeline | cpWER | cpCER | WER (speakers ignored) | CER (speakers ignored) |
|---|---|---|---|---|
| before | 96.0 | 63.6 | 73.2 | 33.0 |
| **after** | **78.2** | **40.3** | **67.9** | **24.8** |
- cpCER 40.3 vs speaker-agnostic 24.8: ~15 points are attribution errors, almost all from 3 clips with a wrong speaker count
  (01: 2→1, 02: 2→4, 09: 3→4). Real recordings (longer turns) show no attribution loss.

### Major limitation + proposed improvement (Task 3C)
**Limitation: clinical vocabulary and Bengali–English code-switching are recognised badly**, the biggest error source on real
consultations (MED CER ~35%, EN ~31%, vs ~21% on read news test speech), and the most harmful one for AIMScribe: a wrong drug name or
dose in a clinical note is a safety issue, while a misspelled filler word is not.
- **Why it happens:** the ASR was fine-tuned on 73 min of OpenSLR read *news* sentences: no drug names, no doctor–patient phrasing,
  English loanwords rarely appear and are spelled inconsistently in Bengali script (টেস্ট/টেষ্ট). Whisper-small's Bengali
  pre-training is thin (zero-shot output was Devanagari). A medical prompt didn't help, because the model was never trained with prompts.
- **Component to modify:** the ASR stage, plus a new post-processing step before role assignment (diarization and roles unchanged).
- **How:**
  1. *In-domain data:* record/collect 5-10 h of Bengali clinical role-play and real consented consultations (doctor + patient
     speakers, phone and clinic mics, fan noise); write English medical terms in one fixed Bengali-script spelling
     (style guide: প্যারাসিটামল, অ্যান্টিবায়োটিক, টেস্ট). Mix with general Bengali (more OpenSLR hours) to avoid forgetting.
  2. *Fine-tune with prompts:* train with a random subset of a medical lexicon as the decoder prompt, so contextual biasing with
     the clinic's drug list works at inference (the prompt test here failed because training never used prompts).
  3. *Lexicon post-correction:* match ASR words against a lexicon of ~2,000 drug/test names common in Bangladesh (with their Bengali
     spellings) using character-level fuzzy matching; replace only above a strict threshold and flag low-confidence matches
     for the doctor to confirm (never silently).
  4. Add noise augmentation (fan, traffic, crowd at 5-20 dB SNR) during fine-tuning (CER rose 39 → 54% at 10 dB here).
- **How to evaluate:** a held-out set of consultations (different speakers and clinics) with tagged medical terms:
  (a) **medical-term recall/precision** (exact drug name + dose correct after normalisation), the primary metric;
  (b) CER/WER overall and on MED/EN-tagged lines (as in `eval_consult.py`) to check nothing else got worse;
  (c) false-correction rate of the lexicon step (correct words wrongly "fixed"); (d) clinician review of a sample of notes.
  Compare before/after on the same held-out set and report per category, as done here.

Secondary limitations (for the report): overlapped speech is transcribed from mixed audio (fix: separate overlapped regions with a
speech-separation model before ASR); speaker counting on very short turns (fix: ask the number of people at the start of the visit;
re-clustering with the known count gave DER 6.8% vs 13.3%); keyword role rules don't understand meaning (fix: an LLM or small
classifier over the whole transcript, with the rule features as input, trained on labelled consultations).

## Phase 4b: stronger ASR checkpoints (2026-10-21)
Question: our whisper-small fine-tune (73 min) is far behind published systems. Are stronger public Bengali checkpoints better on OUR use case?
All zero-shot (no training by us), greedy, 440 tokens, same normalisation. Run on Colab T4 (`evaluate_asr.py`, `eval_consult.py`).

| model | size | OpenSLR test WER / CER | consultations WER / CER | MED CER | EN CER | SHORT CER | OVERLAP CER |
|---|---|---|---|---|---|---|---|
| ours: whisper-small fine-tuned | 244M | 66.2 / 20.7 | 80.9 / 39.2 | 35.0 | 31.3 | 49.2 | 35.3 |
| BanglaASR (whisper-small, ~400 h) | 244M | 36.9 / 9.9 | 61.8 / 27.5 | 22.0 | 18.0 | 45.8 | 41.2 |
| **tugstugi BengaliAI whisper-medium** | 769M | 5.95 / 1.2 ⚠ | **27.2 / 13.0** | **10.8** | **9.4** | **16.2** | 23.5 |
| tugstugi regional whisper-medium | 769M | 14.6 / 2.8 ⚠ | 31.6 / 12.9 | 10.5 | 9.3 | 20.8 | 14.7 |

- ⚠ **Contamination:** tugstugi medium reproduces 88.2% of OpenSLR test sentences exactly (regional 70.3%, ours 10%). Its training data is
  undocumented (empty model card), but this strongly suggests it saw OpenSLR 53. **Its OpenSLR numbers are not reported as a fair result;
  the consultations (recorded by us, never public) are the fair comparison.**
- On the consultations it is genuinely ~3x better than ours, including medical terms (CER 10.8 vs 35.0). Same-size BanglaASR (more data)
  is in between: data volume matters, and a bigger model trained on far more Bengali matters most.
- Our model gives identical consult scores on Colab (transformers 5) and the laptop (transformers 4.46): comparison is like for like.

### Decision rule (fixed BEFORE fine-tuning tugstugi)
- The Task 3 pipeline uses whichever ASR has the lowest **CER on the consultations** (CER, because Bengali word splitting inflates WER).
- Candidate: tugstugi medium + LoRA fine-tune on our 73 min train set (`finetune_whisper.py --lora`, r=32 on q/k/v/out, lr 1e-4,
  batch 4x4, 800 steps, best step by dev WER). Full fine-tuning of 769M params doesn't fit a T4 (~15 GB with AdamW); LoRA used 12.4 GB.
- Caveat stated up front: the train set is probably also in tugstugi's training data, so little gain is expected; the experiment measures
  whether narrow read-news fine-tuning helps or hurts a strong model on conversational clinical speech.
- Whisper-small fine-tune stays as the Task 2 main result (clean, uncontaminated baseline -> fine-tuned comparison).

### Result: LoRA fine-tune of tugstugi medium (300 steps, `results/task2/ft_medium_lora*`)
- **Label-format bug found first:** the first run started at loss ~16. Per-token loss showed two mismatches with how tugstugi was trained:
  (1) it was never trained to *predict* `<|bn|><|transcribe|><|notimestamps|>` (~65 nats/sentence), (2) it writes য়/ড়/ঢ় as one code point,
  our NFC labels as two (য + ়), so it was penalised for its own correct output. Fix (`--match_tugstugi`): prefix tokens fed as decoder
  input but excluded from the loss; labels precomposed. Starting loss 5.36 → **0.09**.
- Loss 0.09 before any training = the model already knows our training sentences almost by heart. Strongest contamination evidence.
- Dev (OpenSLR, contaminated): WER 3.36 → 2.69. **Consultations (fair):**

| tugstugi medium | WER | CER | SHORT CER | EN CER | MED CER | OVERLAP CER |
|---|---|---|---|---|---|---|
| zero-shot | 27.2 | 13.0 | **16.2** | 9.4 | 10.8 | **23.5** |
| + LoRA on our 73 min | 27.6 | 13.0 | 21.7 | 8.8 | 10.7 | 29.4 |

- Tie on CER; short replies (জি/হ্যাঁ/না) got worse (training set = full news sentences only). By the pre-set rule (tie → simpler) the
  pipeline uses **zero-shot tugstugi**. Lesson: fine-tuning a strong model on narrow data it has already seen cannot help and can hurt.

### Result: full pipeline with tugstugi medium (`results/task3/final_tugstugi`, run on Colab in a py3.12 env with the laptop's exact pins)
| pipeline | DER | role error | CER all | CER role-attributed | WER role-attributed |
|---|---|---|---|---|---|
| whisper-small ft (rerun with fixed roles) | 2.66 | 1.18 | 38.6 | 38.5 | 81.2 |
| **tugstugi medium** | **2.66** | **1.18** | **13.0** | **13.1** | **27.9** |
(whisper-small rerun 81.2 vs earlier 80.9 WER: small GPU nondeterminism across runs; CER within 0.2.)

**Role bug exposed by the better ASR:** with tugstugi, consult_B's attendant became "Patient" (role error 46%). Cause: in
"স্যার আমার ছেলেকে নিয়ে এসেছি" the word আমার ("my") was counted as first-person (patient cue), although "আমার ছেলেকে" = my son =
talking about someone else (the docstring already listed it as an attendant cue). The weak ASR had produced an extra "ওর" that hid
the bug, so **the earlier correct result was partly luck**. Fix in `roles.py`: আমার followed by a family word isn't first person.
test_roles.py still 5/5; both pipelines now correct on both recordings. Caveat: fix made after seeing consult_B → validate on held-out
consult_C (Script C in recording_script.md, roles of the two speakers swapped, to be recorded and scored ONCE with everything frozen).
- **Task 3C limitation still holds with the stronger model, and is sharper:** general speech is now mostly right, but drug names
  and doses still break: অ্যামলোডিপিন → "এন্ড্রাডিপিং", মেট্রোনিডাজল → "ব্যাটোনিডাজার", জিংক → "জিন্স", and the dose
  "ফাইভ হান্ড্রেড মিলিগ্রাম" → "ফাইভ হাইড্রেট মিলিগ্রাম". These are exactly the words that matter clinically, which is why the
  proposal (in-domain clinical data + prompt-trained biasing + lexicon post-correction flagged for doctor review) stays the priority.

### Wider model search (2026-10-21): Google / Qwen / DeepSeek + AI4Bharat
Checked open models from the big labs before settling on an ASR (user asked):
- **DeepSeek:** no speech model exists. **Qwen3-ASR** (0.6B/1.7B): 30 languages, Hindi yes, **Bengali not supported**.
  **Qwen3-Omni** (35B): Bengali not among speech-input languages, ~80 GB GPU. → not run.
- **Google Gemma 3n E2B** (open, audio input, "140+ languages", no published Bengali ASR numbers) and
  **AI4Bharat IndicConformer 600M** (22 Indic languages incl. Bengali, MIT) run zero-shot on the consultations, same cuts and scoring
  (`eval_consult.py --backend`, wrappers in `task2_asr/other_asr.py`).

| model (zero-shot, consultations) | WER | CER | SHORT CER | EN CER | MED CER | OVERLAP CER |
|---|---|---|---|---|---|---|
| **IndicConformer 600M, RNN-T** | **18.8** | **7.2** | 14.6 | **6.1** | **6.6** | **17.6** |
| IndicConformer 600M, CTC | 22.8 | 8.5 | 13.3 | 7.4 | 8.1 | 17.6 |
| tugstugi whisper-medium | 27.2 | 13.0 | 16.2 | 9.4 | 10.8 | 23.5 |
| Gemma 3n E2B (fp16) | 1519 | 1115 | – | – | – | – |
| Gemma 3n E2B (bf16 + repetition_penalty 1.2, no_repeat_ngram 3) | 93.8 | 51.4 | 145.0 | 26.1 | 25.6 | 82.4 |
- Gemma 3n fp16 output: repetition loops ("না না না…"), Tamil script, English words ("furnishings"): a general LLM, not an ASR model.
  Even with decoding guards it is far behind; reported as a fair negative result (E4B not tried: needs 4-bit to fit a T4).
- IndicConformer was trained on Indic speech only (IndicVoices etc.), our consultations are private recordings → fair, uncontaminated.
  It has no language-model decoder, so it can't "invent" fluent text the way Whisper/Gemma can.
- **Decision (pre-set rule: lowest consultation CER): IndicConformer RNN-T is the pipeline ASR.**
- Engineering notes: (1) ONNX external weights refuse HF-cache symlinks on Colab → load from a `snapshot_download(local_dir=...)` folder;
  (2) its TorchScript STFT preprocessor fails to compile on CUDA under torch 2.5.1 (laptop version) → run the preprocessor on CPU
  (the ONNX network runs on CPU anyway); verified identical line-level scores (7.2% CER) with this change.

### Full pipeline with IndicConformer (`results/task3/final_indicconformer`, Colab py3.12 env = laptop pins + onnxruntime 1.20.1)
| pipeline | DER | role error | CER all | CER role-attr. | WER role-attr. |
|---|---|---|---|---|---|
| whisper-small ft (Task 2 model) | 2.66 | 1.18 | 38.6 | 38.5 | 81.2 |
| tugstugi whisper-medium | 2.66 | 1.18 | 13.0 | 13.1 | 27.9 |
| **IndicConformer 600M (final)** | **2.66** | **1.18** | **6.8** | **6.8** | **18.4** |
- Transcript now gets almost all clinical words right (প্যারাসিটামল, অ্যান্টিবায়োটিক, জিঙ্ক সিরাপ, স্টুল টেস্ট, ডিহাইড্রেশন, এনএস ওয়ান).
  Remaining clinical errors: অ্যামলোডিপিন → "অ্যান্ডোডি পিন", মেট্রোনিডাজল → "ম্যাট্রোনিডাজার", ফাইভ হান্ড্রেড → "ফাইভ হাইড্রেড" (dose!).

**Second role-rule fix (again exposed by better ASR):** with IndicConformer, consult_B's attendant became "Patient" again. Cause: every
symptom word counted for Patient, but an attendant describing the child's illness says many symptom words (better ASR recognises
more of them). The documented design was "first-person symptoms"; the code now counts a symptom for Patient only if আমার/আমি is within
4 words before it (`own_symptoms`). test_roles 5/5; roles correct on both recordings for all three ASR outputs and on reference text.
Honest caveat: two rule fixes were made after looking at consult_B → the held-out consult_C (Script C) is the real test of the rules.

## Phase 5: held-out SYNTHETIC test sets (2026-10-24)
Why: the reviewer's main point is that consult A+B were used to choose models, read errors and fix role rules, so they are
**development data now**, not a test. New test data was written BEFORE any system saw it, systems frozen (no changes after scoring).
- Generator: `data/make_tts_tests.py`, AI4Bharat **Indic Parler-TTS** (Apache 2.0, 6 Bengali voices: Arjun, Aditi, Tapan, Rashmi,
  Arnav, Riya), run on Colab T4 in its own venv (parler-tts pins transformers 4.46.1). ~5 s per clip.
- (a) **Clinical-vocabulary sentence test** `data/tts/tts_test_sentences.tsv`: 100 sentences (50 doctor: 25 drugs+doses, 13 tests,
  6 diagnoses; 25 patient; 15 attendant; 10 short) x 6 voices; `terms` column = every drug/dose/test name -> medical-term accuracy.
- (b) **6 synthetic consultations** `data/tts/consult_scripts.tsv` (88 lines, 4.6 min): S01 patient first; S02 attendant first, silent
  patient; S03 5 speakers (2 attendants, child patient, nurse=Other); S04 elderly patient + daughter; S05 many interruptions/backchannels;
  S06 10 dB low-frequency "fan" noise. Exact RTTM from line placement (Silero regions per clean TTS line, like prepare_diarization.py).
- Caveats: TTS is cleaner and more regular than real speech (optimistic); scripts written by me knowing the rules; TTS may mispronounce
  (60-clip listening check, `listen_check.csv`). Real held-out recordings (Script C) are still the stronger test.
- Bug found while scoring: speaker names with spaces ("Attendant 1") broke the space-separated RTTM -> names use "_" now (labels only).
- evaluate_pipeline.py: `--set` for other reference folders, ref RTTM speakers mapped to roles for the role error, and **cpWER/cpCER**
  (speaker-aware; role-attributed WER merges two attendants). Re-scoring A+B: identical old numbers (tugstugi: DER 2.66, role err 1.18).

### Task 3 on the 6 synthetic consultations (frozen system, scored once; IndicConformer RNN-T ASR in both rows)
| run | DER | role error | CER all | cpWER | cpCER | role-attr. CER | consultations with all roles right |
|---|---|---|---|---|---|---|---|
| before (plain pyannote, no VAD gate) | 23.7 | 30.4 | 6.9 | 42.9 | 31.8 | 46.4 | 2/6 (S02, S05) |
| **after (re-clustering + VAD gate, final)** | **16.1** | **23.6** | **6.8** | **27.1** | **17.8** | **36.2** | **4/6** (S01, S02, S04, S05) |
Per consultation (after): DER S01 10.9, S02 10.8, S03 17.8, S04 12.7, S05 6.7, S06 37.6; speakers found 2/2, 2/2, 4/5, 3/3, 2/2, 2/3.
- The Task 1 improvement holds on unseen data (DER -7.6 points, cpCER -14 points), mostly by removing extra speakers (S01 3->2, S04 swap).
- Roles from the REFERENCE text are correct in 6/6, so the rules are fine when diarization is right; the failures are interactions:
  - **S03 (roles fail, diarization mostly fine):** doctor = one clean cluster (20.3 s), but nurse line + part of the child formed a small
    cluster (2.8 s) whose per-word doctor score won ("আসুন" matches the polite-verb ending); the real doctor said "ওর বয়স কত" about the
    child -> Attendant. Weakness: doctor score is normalised per word, so a tiny cluster can beat the real doctor.
  - **S06 (diarization fails):** two male voices + fan noise -> patient (10.2 s) merged into the doctor cluster, 3 speakers found as 2.
- Not fixed on purpose (would be tuning on the test). Proposed fixes, to validate on NEW data: require minimum evidence (words) for Doctor
  or score by counts not ratios; ask the number of people at the start of the visit (known count: DER 13.3 -> 6.8 in Task 1).

### Task 2 on the synthetic clinical-vocabulary sentence test (600 clips = 100 sentences x 6 voices, 26.6 min; zero-shot, frozen)
| model | WER | CER | MED CER | SHORT CER | plain-Bengali CER | **term accuracy** | term acc. ignoring spaces |
|---|---|---|---|---|---|---|---|
| **IndicConformer 600M RNN-T (pipeline ASR)** | **11.6** | **2.6** | **3.1** | 4.3 | 1.5 | **71.1** | **77.7** |
| tugstugi whisper-medium | 15.7 | 3.7 | 5.1 | **2.4** | **1.2** | 57.9 | 65.6 |
- Same ranking as on the (dev) real consultations, so the model choice holds on unseen data. tugstugi is slightly better on short
  replies and plain Bengali, IndicConformer clearly better on drug/test names.
- **Key finding:** ordinary Bengali is almost solved (CER 1.5%), but even on clean TTS audio **~1 in 4 drug/dose/test names is wrong**
  (term accuracy 71%). This is the Task 3C limitation measured directly, and the target for fine-tuning.
- Most-missed terms (out of 6 voices): সালবিউটামল 10/12, সেফিক্সিম, মেটফরমিন, ইসোমিপ্রাজল, মন্টিলুকাস্ট, ডমপেরিডন, অ্যাটোরভাস্টাটিন,
  মক্সিফ্লক্সাসিন, এইচবিএওয়ানসি 6/6. Error types: wrong first sound (সেফিক্সিম->তেফিক্সিম, ডমপেরিডন->কম্পেরিডন), word split
  (অ্যাটোর ভাস্টাটিন, এইচবি এ ওয়ানসি), spelling variant (মেটফরমিন->মেটফর্মিন: arguably correct, so strict term accuracy is a lower bound).
  tugstugi also loses the dose: ফাইভ হান্ড্রেড missed 10/12.
- Scoring bug found and fixed BEFORE comparing models: v1 required whole-word matches, so correct inflected terms (থাইরয়েডের,
  মাইগ্রেনের) were counted as misses (64.0 -> 71.1 for IndicConformer). Now a term counts if it starts a word (suffixes allowed).
- Not yet done: human listening check of 60 random clips (`data/tts/listen_check.csv`) -> exclude clips where the TTS itself said the
  wrong thing, then re-score. Audio is on Colab only (/content/aimscribe_data), not yet on Drive/laptop.

## Phase 6: real multi-speaker eval set from a public dataset (2026-10-25)
Decision: the brief tests technical ability, not research volume. So: one public, real, human-labelled multi-speaker set,
prepared as a proper eval (dev for tuning, test scored once), then the overlap problem fixed with the best available method.

### Data: Indic DiarBench, Bengali (sarvamai/indic-diarbench, CC BY 4.0, released 2026-07, test-only benchmark)
- 104 clips, 12.6 h, real near-field meetings / far-field / in-the-wild, 2-9 speakers, human-corrected speaker-attributed
  transcripts with timestamps. Published Bengali cpWER: Sarvam 35, AWS 37, Azure 50, Gemini 3 Pro 54 (pyannote not evaluated).
- `data/prepare_diarbench.py` (mine): RTTM + per-speaker transcripts + explicit overlap-region RTTMs; split **by recording**
  (clips of one recording share speakers/room), stratified by condition: dev 48 clips / 29 recordings / 3.3 h,
  test 56 clips / 45 recordings / 9.3 h.
- Label checks: 1 segment with end <= start dropped, 22 segments running past the audio end clipped, 1 empty transcript kept.
  Speaker ids are free text ("Speaker 0", "New speaker A", "SPEAKER_06") -> sanitised to RTTM-safe ids (spaces broke RTTM).
- `task1_diarization/evaluate.py --overlap_dir`: DER also scored INSIDE reference overlap regions and outside them (UEM).

### Task 1 on dev (3.3 h), collar 0.25
| run | DER | miss | FA | conf | DER in overlap | DER elsewhere |
|---|---|---|---|---|---|---|
| pyannote 3.1 defaults | **18.3** | 9.0 | 1.1 | 8.2 | **38.6** | 13.4 |
| + my re-clustering, thr 0.6 / 0.7 / 0.8 | 20.8 / 24.4 / 30.5 | 9.0 | 1.1 | 10.6 / 14.2 / 20.1 | 48.8 / 45.4 / 44.7 | 14.1 / 19.3 / 26.7 |
| pyannote clustering thr 0.6 / 0.65 / 0.75 / 0.8 (default 0.7046) | 20.5 / 19.3 / 18.7 / 22.2 | 9.0 | 1.1 | 10.3 / 9.1 / 8.5 / 12.0 | 41.8 / 40.6 / 37.7 / 38.8 | 15.0 / 14.0 / 14.1 / 18.1 |
- **Overlap is the problem on real data:** DER inside overlap regions is ~3x the rest, and missed speech (9%) does not move
  with any clustering setting -> the second voice in overlapped speech is not detected at all (a segmentation problem).
- **My re-clustering does NOT transfer** to real meetings (it was designed for 1-3 s simulated turns, and its "overlapping
  segments must be different speakers" rule mislabels real overlap). Reported as a negative result; not used on DiarBench.
- Bug fixed on the way: recluster.embed crashed on a segment at the very end of a clip (empty crop) -> crop clamped to the
  audio and zero-padded to the 0.4 s minimum.
- Next candidates (overlap-aware): pyannote speech-separation-ami-1.0 (PixIT: joint diarization + separation, trained on AMI
  meetings, runs in pyannote 3.3) and pyannote speaker-diarization-community-1 (pyannote 4). Both gated.

### Overlap-aware candidates on dev (collar 0.25)
| run | DER | miss | FA | conf | DER in overlap | DER elsewhere |
|---|---|---|---|---|---|---|
| pyannote 3.1 (all 48 dev clips) | 18.3 | 9.0 | 1.1 | 8.2 | 38.6 | 13.4 |
| **pyannote community-1** (all 48) | **16.7** | 9.0 | 1.1 | **6.5** | **36.5** | **12.1** |
| *32 clips PixIT finished:* 3.1 | 17.8 | 10.2 | 0.5 | 7.2 | 40.2 | 11.2 |
| community-1 | 16.9 | 10.2 | 0.5 | 6.3 | 38.0 | 11.0 |
| PixIT alone (5-min windows) | 42.0 | **3.7** | 18.4 | 20.0 | 39.4 | 41.9 |
| fusion: community-1 + PixIT overlap | 30.8 | 6.2 | 18.3 | 6.3 | 37.7 | 28.8 |
| fusion + "extend existing turns only" (adjacent 0 s) | 17.6 | 7.5 | 3.7 | 6.5 | **33.7** | 13.1 |
| same, min PixIT overlap 1.0 s | 17.6 | 7.9 | 3.3 | 6.5 | 34.4 | 12.8 |
- community-1 (pyannote.audio 4, own venv) wins overall: better clustering (confusion 8.2 -> 6.5). Its missed speech is
  IDENTICAL to 3.1 (same segmentation model), so it does not recover the second voice either.
- PixIT (speech-separation-ami-1.0) is the only system that hears the second voice (missed 10.2 -> 3.7) but on this Bengali
  data it hallucinates overlap and speakers (FA 18%, confusion 20%). Trained on English AMI meetings: domain mismatch.
  Also slow (~1 h for half of dev on the T4) and ran out of 12 GB RAM on 10-15 min clips -> run in 5-min windows.
- My fusion (`overlap_fusion.py`): keep community-1, add a speaker only where PixIT detects overlap, mapped to community-1
  speakers per window (Hungarian on co-occurrence). Plain version adds 34 min of "overlap" -> FA 18%. With the rule
  "an added piece must extend a turn the base system already gives that speaker" (overlap happens at turn changes):
  **DER inside overlap 38.0 -> 33.7**, missed 10.2 -> 7.5, but FA +3.2 just outside the reference turn ends ->
  overall 16.9 -> 17.6. A real overlap gain that does not pay off in total DER: reported as a trade-off, not adopted.
- **Decision (dev only): community-1 is the Task 1 improvement.** Test (9.3 h) scored once: 3.1 vs community-1.

### TEST (9.3 h, 56 clips, 45 recordings never used for any choice), scored once
| run | DER (c=0.25) | miss | FA | conf | DER in overlap | DER elsewhere | DER c=0 | #spk right |
|---|---|---|---|---|---|---|---|---|
| pyannote 3.1 | 19.8 | 6.4 | 3.1 | 10.3 | 38.2 | 18.0 | 23.0 | 35/56 |
| **pyannote community-1** | **16.7** | 6.4 | 3.1 | **7.3** | **35.0** | **15.0** | **19.9** | **46/56** |
By condition (DER c=0.25): far-field 33.3 -> **24.5**, near-field 17.2 -> 17.0, in-the-wild 8.4 -> 8.8.
- Same direction and size as on dev (18.3 -> 16.7): -3.1 points / -16% relative, from better speaker assignment and
  speaker counting (35 -> 46 of 56 clips get the right number of speakers), biggest on far-field audio.
- Missed speech and false alarm are identical: both use the same segmentation model, so the overlap problem
  (DER inside overlap ~2x the rest) remains. Separation (PixIT) is the right direction but needs in-language data.

## Phase 7: clinical fine-tune of IndicConformer 120M (2026-10-25)
Goal: a model I trained, aimed at the measured weakness (drug names 71%), usable in the pipeline if it beats the 600M on dev.
Rule fixed BEFORE training: replaces the 600M in the pipeline only if real consult A+B (dev) CER <= 7.2.

### Training data (`data/make_tts_train.py`)
- 1500 template sentences (drug + dose + frequency/timing/duration, tests, diagnoses, procedures, symptoms, short replies),
  no sentence equal to any test sentence; DEV words (consult A/B, Script C) never appear; test drug/test/diagnosis names split
  40 SEEN (may appear) / 39 UNSEEN (never appear) - asserted in code. Doses are combined at random (fine for hearing words).
- Speech: **AI4Bharat IndicF5** (MIT) cloning 64 real OpenSLR TRAIN speakers -> different TTS model AND voices from the test
  (Parler named voices). Generation stopped at 1058 clips for time (about 3-9 clips/min on the T4; 16 instead of 32 flow
  steps after the first 69 clips). QC: 600M transcribes each clip, CER > 50% dropped (46, 4%: broken TTS) -> 1012 clips,
  81.4 min, 600M CER 6.6% on the kept clips. Mixed 1:1 with real OpenSLR train speech (forgetting guard) = 2024 clips.
- Install notes: IndicF5 needed transformers 4.49, pyarrow 17, datasets 3.1 pins; model must be moved to GPU explicitly.

### Training (`task2_asr/finetune_indicconformer.py`, AI4Bharat NeMo fork, Colab T4)
- RNN-T loss could not be trained here: the numba CUDA kernel can't attach to Colab's CUDA 13 driver (numba 0.58) and newer
  numba rejects the fork's kernels; the pure-PyTorch RNN-T loss did not finish one step in 4 min. -> **CTC branch fine-tuned**
  (CTC weight 1, RNN-T loss replaced by a zero stand-in inside the fused joint), decoded with CTC. Encoder shared.
- Fork bugs worked around: unfused validation path broken (log_probs unassigned), load_from_checkpoint can't rebuild the
  multilingual tokenizer (best weights loaded into the in-memory model instead).
- 1500 steps, batch 8, lr 5e-5 cosine, ~2.5 it/s. Dev = 6 held-out OpenSLR speakers: WER 17.8 (zero-shot) -> **15.5**
  (best, step 750). Real speech improved, not forgotten.

### Results
| dev: real consult A+B (CTC) | WER | CER | MED | EN | SHORT | OVERLAP |
|---|---|---|---|---|---|---|
| 120M zero-shot | 28.3 | 11.1 | 12.9 | 12.1 | 12.9 | 23.5 |
| **120M fine-tuned** | **23.5** | **9.4** | **8.7** | **8.2** | 15.8 | **14.7** |
| (600M RNN-T, pipeline) | 18.8 | 7.2 | 6.6 | 6.1 | 14.6 | 17.6 |

| test: 600 clinical sentence clips | CER | MED CER | plain CER | term acc | seen terms | unseen terms |
|---|---|---|---|---|---|---|
| 120M zero-shot (CTC) | 4.6 | 5.9 | 2.3 | 51.6 | 50.9 | 44.1 |
| **120M fine-tuned (CTC)** | **3.6** | **4.0** | 3.0 | **70.7** | **69.7** | **61.7** |
| 600M RNN-T (pipeline) | 2.6 | 3.1 | 1.5 | 71.1 | 63.6 | 67.1 |
| tugstugi whisper-medium | 3.7 | 5.1 | 1.2 | 57.9 | 56.1 | 50.9 |
- Fine-tuning: drug/test-name accuracy **+19 points** (51.6 -> 70.7), **+17.6 on names never in the training text** (it learnt
  how English drug names are spelled in Bengali, not just a list). Real consultations improved too (CER 11.1 -> 9.4; medical
  lines 12.9 -> 8.7), i.e. the synthetic data transferred to real speech.
- Cost: short replies got worse (12.9 -> 15.8; few short clips in training) and plain sentences slightly (2.3 -> 3.0).
- vs the 600M: better on SEEN drug names (69.7 vs 63.6), worse on unseen and on dev CER (9.4 vs 7.2).
- **Decision by the pre-set rule: dev CER 9.4 > 7.2 -> the pipeline keeps the 600M.** The fine-tune is the Task 2 / 3C result:
  in-domain (even synthetic) clinical data is what moves drug-name accuracy, which supports the proposal (real data + this recipe).

## Phase 8: laptop reproduction of the final system (2026-10-25)
Colab runtime was reset; results that existed only there were regenerated on the laptop (RTX 3050, same code, same models).
| Run | What | Result |
|---|---|---|
| `results/task2/zs_indicconformer600m_consult` | 600M RNN-T on consult A+B lines (main env, onnxruntime CPU) | CER 7.2 / WER 18.8, identical to Colab |
| `results/task3/final_indicconformer` | pipeline: 3.1 + re-clustering + 600M + VAD gate | DER 2.66, role error 1.18, role-attr. WER/CER 18.4 / 6.8, identical to Colab |
| **`results/task3/final_c1`** | **final pipeline: community-1 (pyannote.audio 4.0.7, `.venv_p4`, CPU) + 600M + VAD gate** | **DER 2.66, role error 1.18, roles 2/2, role-attr. WER/CER 18.4 / 6.8, cpCER 6.8** |
- community-1 and 3.1 + re-clustering give the same DER on these 2 clean 2-speaker recordings; they differ on DiarBench (real
  meetings) and the synthetic consultations, which is where the choice was made.
- Speed: pyannote 4 on the laptop CPU takes 2.3x real time (pyannote 3.1 on the laptop GPU: 0.5x). On a Colab T4 the final pipeline runs at ~0.9x.
- Fixes found while doing this: pipeline.py crashed when printing Bengali to a Windows console (cp1252) -> stdout reconfigured to
  UTF-8; pyannote.audio 4.0.1 cannot load its checkpoints under torch >= 2.6 (weights_only) -> pinned 4.0.7.
- Removed stale whisper-era outputs from the root of results/task3 (duplicates of results/task3/final).
- Task 1 deliverable "sample audio with reference and predicted annotations" made complete: `data/make_diarization_samples.py`
  -> `data/samples/diarization/<sample>/` (audio, reference + overlap RTTM, predicted RTTM before/after, timeline plot) for 4 DiarBench
  test clips (012, 097, 072, 013; re-downloaded by streaming), bn_conv_00/09, consult A/B, synthetic S05. DER recomputed from the saved
  predictions matches results/task1 and results/task3 exactly. bengali_013 (improvement hurts) kept on purpose.
- Noise robustness made reproducible: `eval_consult.py --noise_snr` (white noise per line, seed 0). whisper-small ft, 440 tokens:
  CER clean 39.2 -> 20 dB 44.0 -> 10 dB 54.4 -> 5 dB 61.6 (`results/task2/ft_consult_tok440_noise*db`); the original ad-hoc
  measurement was 43.0 / 54.1 / 61.0.
- Synthetic test audio (600 sentence clips, 6 consultations) copied into the repo (`data/tts/audio`, `data/tts_consultation/audio`)
  so the notebooks score exactly the audio that was reported (Parler-TTS sampling is not bit-identical across GPUs).
