"""Task 3: edge-case tests for the role rules (the cases the assessment names).

The two recordings cover "doctor first" and "attendant first, silent patient". These
text-level scenarios cover the rest. Each speaker's utterances are given in the order spoken,
so "who speaks first" is visible in the data, but the rules never use it.

  1. patient speaks first
  2. two attendants + patient + doctor
  3. silent patient, two attendants (mother and father of a child)
  4. a speaker with almost no speech (nurse saying one word) -> Other
  5. noisy ASR-like spelling of the same consultation (fuzzy matching)

Usage: python task3_pipeline/test_roles.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roles import assign_roles  # noqa: E402

SCENARIOS = {
    "1. patient speaks first": (
        {"S1": ["স্যার আমার তিন দিন ধরে জ্বর আর মাথা ব্যথা", "না কোনো ওষুধ খাইনি", "ঠিক আছে ধন্যবাদ স্যার"],
         "S2": ["কত জ্বর ছিল মেপেছিলেন", "একটা সিবিসি টেস্ট করাবেন", "প্যারাসিটামল দিনে তিনবার খাবেন"]},
        {"S1": "Patient", "S2": "Doctor"}),
    "2. two attendants + patient + doctor": (
        {"S1": ["স্যার আমার আব্বাকে নিয়ে এসেছি উনার বুকে ব্যথা"],
         "S2": ["রাতে উনার অনেক ঘাম হচ্ছিল", "উনি ডায়াবেটিসের ওষুধ খান"],
         "S3": ["আমার বুকে ব্যথা হাঁটলে বেশি হয়", "আমি মেটফরমিন খাই"],
         "S4": ["আপনি বসুন ব্যথাটা কবে থেকে", "এখনই একটা ইসিজি টেস্ট করাবেন", "রিপোর্ট দেখে বলব"]},
        {"S1": "Attendant", "S2": "Attendant", "S3": "Patient", "S4": "Doctor"}),
    "3. silent patient, two attendants": (
        {"S1": ["স্যার আমার ছেলের গতকাল থেকে পাতলা পায়খানা", "ওর জ্বরও আছে"],
         "S2": ["বাচ্চাটা রাতে বমি করেছে", "ওকে স্যালাইন খাওয়াইছি"],
         "S3": ["বাচ্চার বয়স কত", "প্রতিবার পায়খানার পর ওরস্যালাইন খাওয়াবেন", "জিংক সিরাপ দশ দিন খাওয়াবেন"]},
        {"S1": "Attendant", "S2": "Attendant", "S3": "Doctor"}),
    "4. speaker with almost no speech": (
        {"S1": ["আপনার কি সমস্যা বসুন", "টেস্ট করাবেন রিপোর্ট দেখাবেন"],
         "S2": ["আমার কাশি আর জ্বর", "আমি কিছু খাইনি"],
         "S3": ["জি"]},
        {"S1": "Doctor", "S2": "Patient", "S3": "Other"}),
    "5. noisy ASR spelling (consult_A style)": (
        {"S1": ["অনোসুদ খেয়েছেন", "এখন দেংগুরসীজন এক্ঠা চিভিসিরা টেষ্ট করাবান", "জড়ের জন্য পেরাসি তেমল মিলিগ্রাম খাবেন"],
         "S2": ["মালেকম আস্যালাম ডাক্টশায়ে আমার তিন দিন জ্বর", "প্রেশারিত হায় থাকেলে আমি খাই", "ঠিকাছে ধোন্যাভাদ"]},
        {"S1": "Doctor", "S2": "Patient"}),
}
SPEECH_S = {"S1": 10, "S2": 10, "S3": 10, "S4": 10}

# Expected to FAIL: shows the limit of keyword rules (they count words, not meaning).
KNOWN_WEAKNESS = {
    "6. patient talks about family, attendant talks about own worry": (
        {"S1": ["আমার ছেলে বলল ডাক্তার দেখাতে", "ওর কথায় এসেছি", "আমার পেটে ব্যথা"],
         "S2": ["আমার খুব ভয় লাগছে স্যার", "আমি সারারাত জেগে ছিলাম"],
         "S3": ["ব্যথাটা কোথায় দেখান", "একটা আলট্রাসনোগ্রাম টেস্ট করাবেন"]},
        {"S1": "Patient", "S2": "Attendant", "S3": "Doctor"}),
}


def main():
    passed = 0
    for name, (texts, expected) in SCENARIOS.items():
        speech = {s: (0.5 if texts[s] == ["জি"] else SPEECH_S[s]) for s in texts}
        roles, feats = assign_roles(texts, speech)
        ok = roles == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {roles}" + ("" if ok else f"   expected {expected}"))
        if not ok:
            for s, f in feats.items():
                print(f"        {s}: doctor={f['doctor_score']:.2f} patient={f['patient_score']} attendant={f['attendant_score']}")
    print(f"\n{passed}/{len(SCENARIOS)} scenarios passed")
    print("\nKnown weakness (expected to fail):")
    for name, (texts, expected) in KNOWN_WEAKNESS.items():
        roles, _ = assign_roles(texts, {s: 10 for s in texts})
        print(f"{'unexpectedly PASSES' if roles == expected else 'FAILS as expected'}  {name}: got {roles}, truth {expected}")


if __name__ == "__main__":
    main()
