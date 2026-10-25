"""Task 3: assign clinical roles (Doctor / Patient / Attendant / Other) to diarized speakers.

Roles are decided from how each speaker talks over the WHOLE consultation, never from
who speaks first (the patient or an attendant often opens the conversation).

Cues (Bengali, matched fuzzily because ASR output is noisy: CER ~40% on real speech):
  Doctor     polite 2nd-person verbs (খাবেন, করাবেন, বসুন, খেয়েছেন), instructions and clinical
             words (টেস্ট, রিপোর্ট, মিলিগ্রাম, দিনে, সিরাপ)
  not Doctor addresses the doctor (স্যার, ডাক্তার সাহেব) or thanks (ধন্যবাদ)
  Attendant  talks ABOUT someone else: ওর, উনার, উনি, ওকে, আমার ছেলে / মেয়ে / আব্বা / মা, বাচ্চা
  Patient    1st-person symptoms: আমার + জ্বর / ব্যথা / কাশি…, আমি … খাই
  Other      too little speech to judge (< MIN_SPEECH_S)

Decision:
  1. Doctor = speaker with the highest doctor score (if it's clearly positive).
  2. Each remaining speaker: Attendant if "about someone else" cues outweigh first-person
     symptom cues, else Patient. Several attendants are allowed; if two speakers both look
     like Patient, the one with more first-person symptom cues keeps Patient and the other
     becomes Attendant (one patient per consultation).
  3. Speakers with very little speech -> Other.
These rules are hand-written from two role-play scripts; they are a transparent baseline,
not a trained model (see the proposal in the report for an LLM / classifier approach).
"""
import re
from difflib import SequenceMatcher

MIN_SPEECH_S = 2.0
FUZZY = 0.75
SELF_WINDOW = 4   # a symptom counts as the speaker's own if আমার/আমি is at most this many words before it

DOCTOR_WORDS = ["টেস্ট", "রিপোর্ট", "মিলিগ্রাম", "দিনে", "তিনবার", "সিরাপ", "ওরস্যালাইন", "স্যালাইন", "হাসপাতালে",
                "প্যারাসিটামল", "অ্যান্টিবায়োটিক", "ইসিজি", "সিবিসি", "খাওয়ার", "বসুন", "বসান", "দরকার", "দিচ্ছি",
                "ডিহাইড্রেশন", "ভাইরাল", "বয়স"]
POLITE_2ND = re.compile(r"(বেন|ছেন|য়েছেন|ুন|বান|চ্ছেন|লেন)$")       # খাবেন করাবেন খেয়েছেন বসুন …
ADDRESS_DOCTOR = ["স্যার", "ডাক্তার", "সাহেব", "ধন্যবাদ"]
ABOUT_OTHER = ["ওর", "উনার", "উনি", "ওকে", "ছেলে", "ছেলেকে", "ছেলের", "মেয়ে", "মেয়েকে", "আব্বা", "আব্বাকে", "আম্মা",
               "বাবা", "মা", "বাচ্চা", "বাচ্চার", "স্বামী", "বউ"]
SYMPTOMS = ["জ্বর", "ব্যথা", "কাশি", "বমি", "পায়খানা", "মাথা", "মাথাটা", "দুর্বল", "শ্বাসকষ্ট", "প্রেশার", "ঘাম"]
FIRST_PERSON = ["আমার", "আমি", "আমাকে"]


def _hit(word, vocab):
    return any(word == v or (len(v) >= 3 and SequenceMatcher(None, word, v).ratio() >= FUZZY) for v in vocab)


def speaker_features(texts):
    """texts: list of the speaker's utterances (normalised Bengali) -> cue counts."""
    words = " ".join(texts).split()
    n = max(1, len(words))
    # "আমার ছেলেকে" (my son) is talk about someone else, not about oneself: আমার followed by a
    # family word is not counted as first person (the family word already counts as about_other)
    self_idx = [i for i, w in enumerate(words)
                if _hit(w, FIRST_PERSON) and not (i + 1 < len(words) and _hit(words[i + 1], ABOUT_OTHER))]
    symptom_idx = [i for i, w in enumerate(words) if _hit(w, SYMPTOMS)]
    f = {
        "words": len(words),
        "doctor_words": sum(_hit(w, DOCTOR_WORDS) for w in words),
        "polite_2nd": sum(bool(POLITE_2ND.search(w)) for w in words),
        "address_doctor": sum(_hit(w, ADDRESS_DOCTOR) for w in words),
        "about_other": sum(_hit(w, ABOUT_OTHER) for w in words if len(w) >= 2),
        "symptoms": len(symptom_idx),
        "first_person": len(self_idx),
        # attendants describe symptoms too (the child's diarrhoea), so only symptoms the speaker claims as
        # their own count for Patient: আমার/আমি shortly before the symptom word ("আমার তিন দিন ধরে জ্বর")
        "own_symptoms": sum(any(0 < s - i <= SELF_WINDOW for i in self_idx) for s in symptom_idx),
    }
    f["doctor_score"] = (f["doctor_words"] + f["polite_2nd"] - 2 * f["address_doctor"]) / n
    # first-person symptom talk vs talking about someone else
    f["patient_score"] = f["first_person"] + 0.5 * f["own_symptoms"]
    f["attendant_score"] = 1.5 * f["about_other"]
    return f


def assign_roles(speaker_texts, speaker_speech_s):
    """speaker_texts: {spk: [utterance texts]}, speaker_speech_s: {spk: seconds}
    -> ({spk: role}, {spk: features}) ."""
    feats = {s: speaker_features(t) for s, t in speaker_texts.items()}
    roles = {}
    active = [s for s in feats if speaker_speech_s.get(s, 0) >= MIN_SPEECH_S]
    for s in feats:
        if s not in active:
            roles[s] = "Other"
    if active:
        doc = max(active, key=lambda s: feats[s]["doctor_score"])
        if feats[doc]["doctor_score"] > 0:
            roles[doc] = "Doctor"
    rest = [s for s in active if s not in roles]
    for s in rest:
        roles[s] = "Attendant" if feats[s]["attendant_score"] > feats[s]["patient_score"] else "Patient"
    patients = [s for s in rest if roles[s] == "Patient"]
    if len(patients) > 1:                       # one patient per consultation
        keep = max(patients, key=lambda s: feats[s]["patient_score"])
        for s in patients:
            if s != keep:
                roles[s] = "Attendant"
    return roles, feats
