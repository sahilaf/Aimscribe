# How this prototype compares with published Bengali results (checked 2026-10-21)

**Caveat:** no public benchmark matches this use case (Bengali doctor–patient consultations with diarization + ASR + roles).
Every published number below uses a different test set, domain, normalisation and amount of training data, so
the comparison shows *where we stand*, not a head-to-head ranking. Our own test sets are small (18 min read speech,
2.2 min real role-play, 13 min simulated conversations).

## 1. Speech recognition (ASR)

| System | Training data | Test data / domain | WER % | CER % |
|---|---|---|---|---|
| **Ours: whisper-small fine-tuned** | **73 min** OpenSLR 53 | OpenSLR 53 read speech, 15 unseen speakers | **66.4** (66.2 with 440 tokens) | **21.4** (20.7) |
| **Ours: same model** (440 tokens) | 73 min | our real role-play consultations | **80.9** | **39.2** |
| BanglaASR whisper-small (zero-shot, run by us) | ~400 h | our consultations | 61.8 | 27.5 |
| **AI4Bharat IndicConformer 600M, RNN-T (zero-shot, run by us; used in final pipeline)** | Indic speech (22 languages) | our consultations | **18.8** | **7.2** |
| Google Gemma 3n E2B (zero-shot, run by us; best decoding setting) | general multimodal LLM | our consultations | 93.8 | 51.4 |
| tugstugi BengaliAI whisper-medium (zero-shot, run by us) | undocumented, large | our consultations | 27.2 | 13.0 |
| same, + our LoRA fine-tune (73 min) | | our consultations | 27.6 | 13.0 |
| same, zero-shot | | OpenSLR 53 test | 5.95 ⚠ | 1.2 ⚠ |
| Ridoy et al. 2025: whisper-small | ~86 h (OpenSLR + Common Voice 17) | OpenSLR + CV17 held-out | 32.2 | 18.2 |
| Ridoy et al. 2025: whisper-large-v2 | ~86 h | same | 28.9 | 7.5 |
| Ridoy et al. 2025: **Wav2Vec-BERT** | ~86 h | same | **14.4** | **2.7** |
| arijitx: wav2vec2 XLSR-53 (model card) | 40k utts OpenSLR 53 | OpenSLR 53 held-out | 32.5 | – |
| Kabir et al. 2024 (MedAI): **Whisper (BanglaASR) fine-tuned** | **57.6 h medical** Bengali (symptoms, diseases, Sylheti, synthetic, 0.68 h real doctor–patient) | 5.8 h medical held-out | **9.1** (noisy subset 20.2) | – |
| Kabir et al. 2024: DeepSpeech2 | same | same | 17.3 (noisy 41.7) | – |
| Bengali-Loop 2026: tugstugi whisper-medium (pretrained Bengali) | community Bengali fine-tune | 158 h long-form drama/news (YouTube) | 34.1 | 16.4 |
| Bangla-WhisperDiar 2026: whisper-medium fine-tuned | Bengali-Loop train (~140 h) | Bengali-Loop test | 24.4 | – |
| ShobdoSetu 2026: whisper-medium fine-tuned | ~21k chunks Bengali-Loop | competition private test / **out-of-domain** | 15.6 / **41.1** | – |

⚠ = likely trained on OpenSLR 53 (88% of test sentences reproduced exactly; loss 0.09 on our training sentences before any training), so
not a fair score; our own recordings are.

**Reading it:**
- With ~70x less data, our CER on read speech (21.4%; 20.7% with the 440-token budget) is close to the published whisper-small trained on ~86 h (18.2%).
  Our WER is twice theirs, largely from word-splitting and spelling variants (একইসঙ্গে → একই সঙ্গে costs 2 words; CER barely moves).
- The real gap is **model size, data and domain**. Larger Whisper models or Wav2Vec-BERT with 50-150 h reach 9-25% WER, and
  the medical system (Kabir et al.) reaches 9.1% WER with **57.6 h of in-domain medical speech**. Our real-consultation
  WER (80.9%) reflects a model that has never heard clinical vocabulary or conversational speech.
- Even strong systems collapse out of domain (ShobdoSetu: 15.6 → 41.1% WER; noisy medical audio: 9.1 → 20.2%), which supports our
  proposal: in-domain clinical data matters more than a bigger model.

## 2. Speaker diarization

| System | Test data | Speakers / recording | DER % |
|---|---|---|---|
| **Ours, final: pyannote community-1** (chosen on dev) | **Indic DiarBench Bengali, my test split (9.3 h real meetings / far-field / in-the-wild, human labels)** | 2-8 | **16.7** (pyannote 3.1: 19.8) |
| **Ours: pyannote 3.1 + segment re-clustering** | our 2 real role-play recordings (2.2 min) | 2 | **2.7** (pyannote alone 7.4) |
| **Ours: same** | 10 simulated Bengali conversations (13 min), test half | 2-3 | **15.9** (pyannote alone 28.1); 7.4 with known count |
| Bengali-Loop 2026: pyannote pretrained | 12.5 h Bengali drama, manual labels | 1-27 | 40.1 |
| Bangla-WhisperDiar 2026: fine-tuned pyannote segmentation | Bengali-Loop test | 1-27 | 23.9 |
| ShobdoSetu 2026: fine-tuned pyannote community-1 segmentation | competition private test | many | 26.7 |
| Baroudi et al. 2026: EEND-VC on w2v-BERT 2.0 (**doctor–patient, code-switched Hinglish**) | medical consultations | 2+ | **7.8** (from 9.3) |

**Reading it:**
- Bengali-Loop is far harder than a consultation (drama with 10-27 speakers, music, overlapping dialogue), so our 2.7-15.9% doesn't
  mean we beat it. The closest use case, doctor–patient Hinglish, reports 7.8%. Our real-recording figure (2.7%) is in that
  range but comes from only 2.2 min with references that started from system boundaries (likely optimistic).
- The published gains come from **fine-tuning the segmentation model** on in-language data. Our re-clustering (training-free)
  helped on simulated short turns but not on real DiarBench meetings (dev 18.3 → 20.8+), so the final system uses community-1
  without changes. The remaining error is concentrated in overlap (DER 35 % inside overlap vs 15 % elsewhere), which is
  what fine-tuning segmentation on Bengali overlap would target.
- (Update 2026-10-25: the DiarBench rows were added after this comparison was first written.)

## 3. Speaker-attributed transcript (the full Task 3 output)

| System | Metric | Score |
|---|---|---|
| **Ours** (whisper-small pipeline) | cpWER / cpCER, 10 simulated conversations | 78.2 / 40.3 (no improvements: 96.0 / 63.6) |
| **Ours** (whisper-small fine-tuned) | role-attributed WER / CER, real role-play | 81.2 / 38.5 |
| Ours (tugstugi whisper-medium) | role-attributed WER / CER, real role-play | 27.9 / 13.1 |
| **Ours, final** (IndicConformer 600M) | role-attributed WER / CER, real role-play | **18.4 / 6.8** |
| Baroudi et al. 2026 (Hinglish doctor–patient; Qwen3-ASR + EEND-VC + LLM correction) | tcpWER | 18.6 (from 26.8) |

No published Bengali system reports speaker-attributed WER or **doctor/patient role accuracy**. Ours (2/2 real recordings correct,
5/5 rule unit cases plus 1 documented failure mode, 3/6 synthetic consultations fully right) is a first transparent baseline rather than a comparison point.

## 4. What closing the gap would take (ordered by expected impact per effort)
1. **Start from a stronger Bengali checkpoint** (DONE): of six candidates (incl. Google Gemma 3n; Qwen3-ASR and DeepSeek have no
   Bengali speech model), AI4Bharat IndicConformer cut the pipeline's consultation CER 38.6 → 6.8% (ASR alone: 39.2 → 7.2%). Fine-tuning tugstugi whisper-medium on OpenSLR
   gave no gain (data it already knows), so the next gain needs new, in-domain clinical data.

Sources for the model search: [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (language list),
[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), [Gemma 3n](https://huggingface.co/google/gemma-3n-E4B-it),
[IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), SraVaani 1.0 (arXiv 2608.08235: IndicConformer
22.7% WER vs Gemini 3 Flash 24.7% on 32 h Bengali).
2. **More and in-domain data**: tens of hours (OpenSLR 53 has ~229 h; Common Voice Bengali), plus medical speech
   (Kabir et al.'s results show the value of medical data; MediBeng offers synthetic Bengali–English clinical code-switching).
3. **Noise augmentation** (the medical paper's noisy subset doubles WER; ours rose 39 → 54% CER at 10 dB SNR).
4. **Fine-tune the pyannote segmentation model** on Bengali conversation, and keep our re-clustering on top.

## Sources
- Bengali-Loop (Tabib et al., arXiv 2602.14291, Feb 2026): https://arxiv.org/html/2602.14291
- Bangla-WhisperDiar (Bhuiyan et al., arXiv 2605.08214, May 2026): https://arxiv.org/html/2605.08214v1
- ShobdoSetu (Tanvir et al., arXiv 2603.19256, Mar 2026): https://arxiv.org/pdf/2603.19256
- Ridoy et al., Whisper vs Wav2Vec-BERT for Bangla (arXiv 2507.01931, Jul 2025): https://arxiv.org/html/2507.01931
- Kabir et al., ASR for Biomedical Data in Bengali (arXiv 2406.12931, 2024): https://arxiv.org/pdf/2406.12931
- Baroudi et al., Doctor or Patient? Diarization + ASR for code-switched Hinglish medical (arXiv 2603.06373, 2026): https://arxiv.org/abs/2603.06373
- arijitx/wav2vec2-large-xlsr-bengali model card: https://huggingface.co/arijitx/wav2vec2-large-xlsr-bengali
- MediBeng (synthetic Bengali–English clinical code-switching): https://www.medrxiv.org/content/10.1101/2025.04.25.25326406v1.full.pdf
- Chowdhury & Chowdhury, long-form Bangla ASR + diarization (arXiv 2602.21741): https://arxiv.org/html/2602.21741v1
