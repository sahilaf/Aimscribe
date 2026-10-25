# Role-Play Consultation Recording Script

These recordings are the **realistic test audio for Task 3** (and extra test cases for Task 2's error analysis).
Each script is built to trigger the specific problems the assessment asks about:

| Tag | Tests | Assessment item |
|---|---|---|
| 🔴 **OVERLAP** | two people talking at once | SD overlap, Task 3 roles |
| 🟡 **SHORT** | very short replies (জি / হ্যাঁ / না / হুম) | SD missed short utterances, ASR short words |
| 🔵 **EN** | English words inside Bengali | ASR code-switching |
| 🟢 **MED** | medicine / test / disease names | ASR medical terminology |
| ⚪ **PAUSE** | 2-3 s of silence | ASR hallucination on silence |

---

## Recording setup

1. **Consent:** both speakers agreed to the recordings being used and shared.
2. **Device:** one phone on the table between the speakers (about 50 cm from each), normal voice-recorder app (no noise suppression).
3. **Room:** quiet room.
4. Natural speech, not read aloud. Changes from the script are written into the reference transcript.
5. 2 s of silence at the start and end; each person stays in the same seat.
6. Originals: `audio/original/consult_A.m4a`, `consult_B.m4a`.

---

## Script A: Fever consultation (2 people, about 2 minutes)
**Doctor (D)** and **Patient (P)**. The doctor speaks first; this is the "normal" case.

| # | Who | Line | Tags |
|---|---|---|---|
| A1 | D | আসসালামু আলাইকুম, বসুন। বলুন, আপনার কী সমস্যা হচ্ছে? | |
| A2 | P | ওয়ালাইকুম আসসালাম। ডাক্তার সাহেব, আমার তিন দিন ধরে জ্বর। | |
| A3 | D | জ্বর কত ছিল? মেপেছিলেন? | |
| A4 | P | জি, কাল রাতে একশো দুই ছিল। | 🟡 |
| A5 | D | হুম। সাথে কাশি বা গলা ব্যথা আছে? | 🟡 |
| A6 | P | হ্যাঁ, একটু কাশি আছে। আর মাথাটা খুব ব্যথা করে। | 🟡 |
| A7 | D | কোনো ওষুধ খেয়েছেন? | |
| A8 | P | না, এখনো খাইনি। শুধু গরম পানি খাচ্ছি আর— | 🟡 |
| A9 | D | *(start talking while P is still finishing A8)* প্যারাসিটামলও খাননি? | 🔴 🟢 |
| A10 | P | না। | 🟡 |
| A11 | D | আপনার কি ডায়াবেটিস বা হাই ব্লাড প্রেশার আছে? | 🔵 🟢 |
| A12 | P | প্রেশার একটু হাই থাকে। আমি অ্যামলোডিপিন ফাইভ মিলিগ্রাম খাই। | 🔵 🟢 |
| A13 | D | ঠিক আছে। শ্বাসকষ্ট হয়? | |
| A14 | P | না। | 🟡 |
| A15 | D | বমি বা পাতলা পায়খানা হয়েছে? | |
| A16 | P | হুম, গতকাল একবার বমি হয়েছে। | 🟡 |
| — | — | *(doctor "writes" silently for 3 seconds)* | ⚪ |
| A17 | D | আচ্ছা। এখন ডেঙ্গুর সিজন, তাই একটা সিবিসি আর ডেঙ্গু এনএস-ওয়ান টেস্ট করাবেন। | 🔵 🟢 |
| A18 | P | টেস্টগুলো কি আজকেই করাতে হবে? | 🔵 |
| A19 | D | জি, আজকেই করান। রিপোর্ট পেলে আমাকে দেখাবেন। | 🔵 |
| A20 | P | আচ্ছা। | 🟡 |
| A21 | D | জ্বরের জন্য প্যারাসিটামল ফাইভ হান্ড্রেড মিলিগ্রাম, দিনে তিনবার, খাওয়ার পরে। প্রচুর পানি আর ওরস্যালাইন খাবেন। | 🔵 🟢 |
| A22 | P | অ্যান্টিবায়োটিক লাগবে না? | 🟢 |
| A23 | D | না, এখন অ্যান্টিবায়োটিক লাগবে না। মনে হচ্ছে ভাইরাল ফিভার। তবে যদি— | 🔵 🟢 |
| A24 | P | *(say this in the middle of A23, while D keeps talking)* জি। | 🔴 🟡 |
| A25 | D | —পেটে ব্যথা, রক্তপাত বা খুব দুর্বল লাগে, সাথে সাথে হাসপাতালে আসবেন। | |
| A26 | P | ঠিক আছে, ধন্যবাদ ডাক্তার সাহেব। | |
| A27 | D | ভালো থাকবেন। | |

---

## Script B: Child with diarrhoea, attendant speaks first (2 people, about 1.5 minutes)
**Attendant (A)** is a parent who brought their young child, and **Doctor (D)**. The child (the patient) is "in the room" but **never speaks**.

This covers the hard cases from the PDF using only 2 people:
- **The doctor does not speak first.** The attendant opens the conversation.
- **The person describing symptoms is not the patient.** A naive rule ("whoever describes symptoms = Patient") would mislabel the attendant. The role system must notice phrases like *আমার ছেলের*, *ওর*, *বাচ্চাটা* (talking about someone else), so this speaker gets labelled **Attendant**.
- **The patient has no voice at all,** so the system must not invent a Patient speaker.

| # | Who | Line | Tags |
|---|---|---|---|
| B1 | A | স্যার, আমার ছেলেকে নিয়ে এসেছি। ওর গতকাল থেকে পাতলা পায়খানা হচ্ছে। | |
| B2 | D | আচ্ছা, বসুন। বাচ্চার বয়স কত? | |
| B3 | A | চার বছর। | 🟡 |
| B4 | D | দিনে কতবার পায়খানা হচ্ছে? | |
| B5 | A | প্রায় ছয়-সাতবার। আর রাতে একবার বমিও করেছে, তারপর থেকে— | |
| B6 | D | *(cut in while A is finishing B5)* জ্বর আছে? | 🔴 |
| B7 | A | হ্যাঁ, একটু জ্বর আছে। | 🟡 |
| B8 | D | হুম। প্রস্রাব ঠিকমতো হচ্ছে? | 🟡 |
| B9 | A | না, কম হচ্ছে। | 🟡 |
| B10 | D | কিছু খাওয়ানো হয়েছে? কোনো ওষুধ? | |
| B11 | A | ফার্মেসি থেকে একটা সিরাপ দিয়েছিল, মেট্রোনিডাজল মনে হয়। | 🟢 |
| — | — | *(doctor "examines" the child silently for 3 seconds)* | ⚪ |
| B12 | D | ঠিক আছে। ওর একটু ডিহাইড্রেশন হয়েছে। প্রতিবার পায়খানার পর আধা কাপ ওরস্যালাইন খাওয়াবেন। | 🔵 🟢 |
| B13 | A | জি। | 🟡 |
| B14 | D | আর জিংক সিরাপ দিচ্ছি, দশ দিন খাওয়াবেন। অ্যান্টিবায়োটিক এখন দরকার নেই। | 🟢 |
| B15 | A | *(say this in the middle of B14, while D keeps talking)* আচ্ছা। | 🔴 🟡 |
| B16 | A | স্যার, স্টুল টেস্ট করাতে হবে? | 🔵 🟢 |
| B17 | D | এখন না। যদি দুই দিনে ভালো না হয় বা পায়খানার সাথে রক্ত যায়, তখন টেস্ট করাব। | 🔵 |
| B18 | A | ঠিক আছে, ধন্যবাদ স্যার। | |

**Not covered by the recordings: multiple attendants.** You can't record that with 2 people. Handle it in Task 3 by (1) describing in the role-assignment proposal how a 3rd/4th speaker who talks *about* the patient gets the Attendant label, and (2) optionally testing the rule on a 3-speaker simulated clip from `data/diarization/`.

---

## Script C: Stomach pain, HELD-OUT test (2 people, about 1.5 minutes) (not recorded)
**Why a third recording:** every setting in the pipeline (re-clustering threshold, VAD gate, which ASR model) was chosen
by looking at consult_A and consult_B, so their scores are slightly optimistic. **Script C is used once, at the very end,
with everything frozen.** No changes are made after seeing its result. That makes it the honest final number.

Two deliberate changes:
- **Swap who plays whom.** Whoever played the doctor in A and B now plays the **patient**. If the roles still come out right,
  the system decides by *what is said*, not by *whose voice it is*.
- **The patient speaks first** and asks a question, which is the hard case from the PDF.

**Patient (P)** and **Doctor (D)**:

| # | Who | Line | Tags |
|---|---|---|---|
| C1 | P | ডাক্তার সাহেব, আসব? আমার পেটে খুব ব্যথা। | |
| C2 | D | জি আসুন, বসুন। কতদিন ধরে ব্যথা? | 🟡 |
| C3 | P | প্রায় এক সপ্তাহ। খাওয়ার পরে বুক জ্বালাপোড়া করে। | |
| C4 | D | ব্যথাটা ঠিক কোথায়? উপরের দিকে না নিচের দিকে? | |
| C5 | P | উপরের দিকে, নাভির উপরে। | |
| C6 | D | রাতে ব্যথা বাড়ে? | |
| C7 | P | হ্যাঁ। | 🟡 |
| C8 | D | কোনো ওষুধ খাচ্ছেন? | |
| C9 | P | ব্যথার জন্য মাঝে মাঝে আইবুপ্রোফেন খাই, আর গ্যাস্ট্রিকের জন্য— | 🟢 |
| C10 | D | *(cut in while P is finishing C9)* আইবুপ্রোফেন খালি পেটে খান? | 🔴 🟢 |
| C11 | P | জি। | 🟡 |
| C12 | D | ওটাই সমস্যা। আইবুপ্রোফেন এখন বন্ধ রাখবেন। | 🟢 |
| — | — | *(doctor "writes" silently for 3 seconds)* | ⚪ |
| C13 | D | ওমিপ্রাজল টুয়েন্টি মিলিগ্রাম, সকালে নাস্তার আধা ঘণ্টা আগে, চৌদ্দ দিন। | 🔵 🟢 |
| C14 | P | আচ্ছা। | 🟡 |
| C15 | D | আর একটা আল্ট্রাসনোগ্রাম করাবেন, পুরো পেটের। | 🔵 🟢 |
| C16 | P | এন্ডোস্কপি লাগবে না? | 🟢 |
| C17 | D | এখনই না। দুই সপ্তাহে ভালো না হলে বা কালো পায়খানা হলে এন্ডোস্কপি করাব। | 🟢 |
| C18 | P | *(say this in the middle of C17, while D keeps talking)* হুম। | 🔴 🟡 |
| C19 | D | তেল-মশলা কম খাবেন, আর রিপোর্ট নিয়ে আবার আসবেন। | 🔵 |
| C20 | P | ঠিক আছে, ধন্যবাদ। | |

Save as `data/consultation/audio/original/consult_C.m4a` (same phone, same room, same distance as A and B).

---

## From recording to reference labels
1. `labels_to_reference.py` input: Audacity label files (`labels/consult_*.txt`), one label per script line.
2. Draft timestamps (`labels/consult_*_draft.txt`) were made by `prefill_labels.py` (diarization + alignment to the script),
   then every boundary and every line of text was corrected by ear in Audacity.
3. `labels_to_reference.py` writes the reference RTTM (`rttm/`) and transcript (`transcripts/`); the audio is converted to 16 kHz mono WAV (`audio/`).
