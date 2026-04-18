"""
intent_engine.py — Exhaustive Intent Classifier for Simran
============================================================
This file contains:
  1. CLASSIFY_INTENT  — covers every way Indian customers say yes/no/maybe
  2. FAST_TRAINER     — optimizes system prompt automatically (no GPU needed)
  3. PROMPT_OPTIMIZER — finds the best system prompt by testing variants

Philosophy:
  - YES/NO logic lives in Python code, never in the LLM
  - LLM only generates the words — Python makes all decisions
  - classify_intent() must be exhaustive — every Indian customer phrase covered
  - Training = improving the system prompt, not fine-tuning weights

How to use:
  # In main.py, replace your classify_intent() with this one:
  from intent_engine import classify_intent, INTENT_TEST_SUITE
  
  # To optimize your system prompt:
  python intent_engine.py --optimize
  
  # To test all intent cases:
  python intent_engine.py --test
"""

import re
import os
import json
import time
from pathlib import Path
from openai import OpenAI

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

def contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE SIGNAL BANKS
# Every way an Indian customer says yes, no, maybe, busy
# ─────────────────────────────────────────────────────────────────────────────

# ── HARD NO — definitive rejection ───────────────────────────────────────────
# Rule: if customer says any of these, end the call warmly
HARD_NO_DEVANAGARI = [
    # Direct rejection
    "नहीं चाहिए", "नहीं करना", "नहीं लेना", "नहीं जाना",
    "नहीं है", "बिल्कुल नहीं", "कोई जरूरत नहीं", "ज़रूरत नहीं",
    "interest नहीं", "मुझे नहीं", "हमें नहीं",
    # Action commands to stop
    "छोड़ो", "रहने दो", "बंद करो", "मत करो", "call मत करो",
    "phone मत करो", "हटाओ", "remove करो",
    # I won't go / don't want
    "मुझे नहीं जाना", "जाना नहीं है", "जाना ही नहीं है",
    "कहीं नहीं जाना", "नहीं जाना है", "नहीं जाना था",
    # I'm not doing it
    "नहीं कर रहा हूं", "नहीं कर रही हूं", "नहीं करना है",
    "नहीं करूंगा", "नहीं करूंगी",
]

# Sentence patterns that signal hard no (Devanagari context)
HARD_NO_DEVANAGARI_PATTERNS = [
    r"नहीं.{0,15}जाना",       # नहीं ... जाना
    r"कहीं नहीं",              # कहीं नहीं
    r"नहीं.{0,10}करना",       # नहीं ... करना
    r"नहीं.{0,10}चाहिए",      # नहीं ... चाहिए
    r"नहीं.{0,10}interest",   # नहीं ... interest
]

# Future/soft signals — if present WITH नहीं, it is soft_no not hard no
FUTURE_SOFT_DEVANAGARI = [
    "बाद में", "बाद मे", "जाऊंगा", "जाऊंगी", "जाएंगे", "जायेंगे",
    "करूंगा", "करूंगी", "करेंगे", "सोचेंगे", "सोच लेंगे",
    "देखते हैं", "देखेंगे", "पता नहीं", "decide नहीं",
    "ज़रूर जाऊंगा", "ज़रूर जाऊंगी",
]

# Budget words — if present WITH नहीं, it is budget correction not rejection
BUDGET_WORDS_DEVANAGARI = [
    "लाख", "हज़ार", "हजार", "₹", "रुपये", "रुपया",
    "budget", "खर्चा", "पैसे", "राशि",
]

# Hard no regex (Roman/Hinglish/English)
HARD_NO_REGEX = re.compile(
    r"\b("
    # Direct no words
    r"nahi\s*nahi|bilkul\s*nahi|absolutely\s*not|not\s*at\s*all|"
    r"never|no\s*way|nope|"
    # Not interested
    r"not\s*interested|koi\s*zaroorat\s*nahi|zaroorat\s*nahi|"
    r"interest\s*nahi|mujhe\s*nahi|humein\s*nahi|"
    # Don't want / don't call
    r"don'?t\s*want|dont\s*want|don'?t\s*need|dont\s*need|"
    r"don'?t\s*call|dont\s*call|band\s*karo|"
    r"remove|unsubscribe|"
    # Won't go
    r"nahi\s*jaana|nahi\s*jana|jaana\s*hi\s*nahi|jana\s*hi\s*nahi|"
    r"kahin\s*nahi\s*jaana|kahi\s*nahi\s*jaana|"
    # Won't do
    r"nahi\s*karna|nahi\s*kar\s*raha|nahi\s*kar\s*rahi|"
    r"nahi\s*chahiye|nahi\s*chahie|"
    # Polite English no
    r"no\s*thanks|no\s*thank\s*you|not\s*now\s*thanks|"
    r"i'?m\s*good|i'?m\s*fine\s*thanks|"
    # Angry
    r"rehne\s*do|chhodo|chodo|hatao"
    r")\b",
    re.IGNORECASE
)

# ── SOFT NO — hesitation, NOT rejection ──────────────────────────────────────
# Rule: acknowledge, keep conversation open, don't give up
SOFT_NO_SIGNALS = [
    # Will think about it
    "sochenge", "soch lenge", "soch ke batata", "soch ke batati",
    "soch lunga", "soch lungi", "think kar loon", "think karke",
    "dekh lenge", "dekhte hain", "baad mein dekhenge",
    # Not sure yet
    "not sure", "not sure yet", "not sure about", "pata nahi abhi", "pata nahi kahan",
    "decide nahi kiya", "decide nahi hua", "abhi nahi socha",
    "abhi nahi socha", "socha nahi", "plan nahi",
    # Need to discuss
    "wife se poochna", "wife se puchna", "husband se poochna",
    "family se poochna", "ghar mein poochna", "baat karni hai",
    "discuss karna hai", "ghar mein baat",
    # Later
    "baad mein", "baad me", "kal batata", "kal batati",
    "kal baat karte", "thodi der mein", "thoda time chahiye",
    "abhi nahi par", "abhi nahi lekin", "abhi nahi but",
    # Will come / will go later
    "baad mein aaunga", "baad mein jaunga", "baad mein jaaunga",
    "zaroor jaunga", "zaroor jaaunga", "plan hai future mein",
    # Devanagari soft no
    "बाद में जाऊंगा", "बाद में जाऊंगी", "बाद में जाएंगे",
    "अभी नहीं पर", "अभी नही पर", "बाद में देखेंगे",
    "सोचेंगे", "सोच लेंगे", "pata nahi abhi",
    "discuss करना है", "family से पूछना है",
    "wife से पूछना है", "husband से पूछना है",
]

# ── BUSY — wrong time, not rejection ─────────────────────────────────────────
# Rule: apologize and exit, they are genuinely busy
BUSY_DEVANAGARI = [
    "busy", "अभी busy", "बाद में call", "बाद में बात", "थोड़ी देर",
    "meeting", "office", "driving", "गाड़ी चला",
]
BUSY_REGEX = re.compile(
    r"\b("
    r"busy|in\s*a\s*meeting|on\s*a\s*call|driving|"
    r"baad\s*mein\s*call|call\s*back|call\s*later|"
    r"baad\s*mein\s*baat|thodi\s*der\s*mein|"
    r"abhi\s*free\s*nahi|abhi\s*time\s*nahi|"
    r"meeting\s*mein|office\s*mein|gari\s*chala|"
    r"kaam\s*kar\s*raha|kaam\s*kar\s*rahi|"
    r"occupied|not\s*a\s*good\s*time|bad\s*time"
    r")\b",
    re.IGNORECASE
)

# ── YES — interested, proceed ─────────────────────────────────────────────────
# Rule: move to next stage of conversation
YES_DEVANAGARI = [
    "हाँ", "हां", "हा", "बिल्कुल", "जी हाँ", "जी हां",
    "ठीक है", "ठीक", "सही है", "अच्छा", "बताइए",
    "ज़रूर", "हाँ बताइए", "हां बताइए",
]
YES_REGEX = re.compile(
    r"\b("
    r"haan|han|yes|ya\b|yep|yup|yaar\s*haan|"
    r"bilkul|zaroor|sure|of\s*course|"
    r"theek\s*hai|theek\s*he|sahi\s*hai|"
    r"interested|plan\s*hai|plan\s*kar\s*rahe|"
    r"jaana\s*hai|jaana\s*chahte|jana\s*chahta|"
    r"trip\s*plan|vacation\s*plan|travel\s*karna|"
    r"book\s*karna|booking\s*karni|"
    r"chalna\s*hai|chalte\s*hain|"
    r"bata\s*do|bataiye|batao|"
    r"i'?m\s*interested|i\s*want\s*to|i'?d\s*like"
    r")\b",
    re.IGNORECASE
)

# ── FILLER — don't process these ─────────────────────────────────────────────
FILLER_WORDS = {
    "umm", "ummm", "hmm", "hmmm", "uh", "uhh", "ah", "ahh",
    "err", "hm", "mm", "mmm", "oh", "ohh",
}

# ── GREETING — respond warmly before asking destination ──────────────────────
GREETING_ROMAN = {
    "hello", "hi", "hey", "helo", "hii", "heyy", "hlo",
    "namaste", "namaskar", "namasthe",
    "sat sri akal", "adaab", "jai shree krishna", "jai shri ram",
    "good morning", "good afternoon", "good evening",
    "bol", "bolo", "boliye", "suno", "suniye",
    "haan", "han", "ha", "ok", "okay", "theek", "theek hai",
    "yes", "ya", "yep", "yup",
    # "acha", "achha" REMOVED — too ambiguous, classify as unclear
}
GREETING_DEVANAGARI = {
    "हेलो", "हाय", "हे", "नमस्ते", "नमस्कार",
    "हाँ", "हां", "हा", "बोलो", "बोलिए",
    "सुनो", "सुनिए", "ठीक है", "ओके", "जी",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def is_filler_only(text: str) -> bool:
    words = set(text.lower().strip().split())
    return bool(words) and len(words) <= 2 and words.issubset(FILLER_WORDS)

def is_greeting_only(text: str) -> bool:
    if not text: return False
    raw_words   = set(text.strip().split())
    lower_words = set(text.lower().strip().split())
    combined    = raw_words | lower_words
    if not combined or len(raw_words) > 4: return False
    return all(w in GREETING_ROMAN or w in GREETING_DEVANAGARI for w in combined)

def has_budget_signal(text: str) -> bool:
    """Check if text contains any budget/money signal."""
    t = text.lower()
    budget_roman = [
        r"lakh", r"hazaar", r"hazar", r"hajar", r"thousand",
        r"budget", r"rupee", r"rupaye", r"rs", r"₹",
        r"spend", r"kharcha", r"cost", r"price",
        r"affordable", r"cheap", r"sasta", r"mehnga",
    ]
    if any(re.search(r"\b" + b + r"\b", t) for b in budget_roman): return True
    if any(b in text for b in BUDGET_WORDS_DEVANAGARI): return True
    if re.search(r"\b\d{4,6}\b", t): return True
    if re.search(r"\b\d+\s*k\b", t): return True
    return False

def classify_intent(text: str) -> str:
    """
    Exhaustive intent classifier.
    
    Returns one of:
      filler        — "umm", "hmm" — ignore completely
      greeting_only — "hello", "हेलो" — greet back warmly
      no            — hard rejection — exit call warmly
      soft_no       — hesitation — keep conversation open
      busy          — wrong time — apologize and exit
      yes           — interested — proceed to next stage
      has_budget    — gave budget info — save it
      unclear       — anything else — ask again differently
    
    KEY RULE: Budget signal always overrides no signal.
    "नहीं 5 लाख" = customer correcting amount, not rejecting.
    """
    if not text or not text.strip():
        return "unclear"

    raw  = text.strip()
    norm = raw.lower()

    # ── Step 1: Filler — ignore completely ───────────────────────────────
    if is_filler_only(norm):
        return "filler"

    # ── Step 2: Budget override (High priority corrections) ──────────────
    # "नहीं 5 लाख" = customer correcting amount, not rejecting.
    if has_budget_signal(raw):
        has_rejection = ("नहीं" in raw or "नही" in raw or 
                         bool(re.search(r"\b(no|nope|nahi|nahin)\b", norm)))
        if has_rejection:
            return "has_budget"
        # Pure budget statement (Step 7 will catch if not high priority)
        # But for now, we leave it to check Busy/No first

    # ── Step 3: Busy/Wrong Time — High priority exit ──────────────────────
    if BUSY_REGEX.search(norm):
        return "busy"
    if any(p in raw for p in BUSY_DEVANAGARI):
        return "busy"

    # ── Step 4: Soft No (Hesitations override Hard No) ───────────────────
    # "नहीं बाद में देखेंगे" should be soft_no, not no.
    if any(s in norm for s in SOFT_NO_SIGNALS):
        return "soft_no"
    if contains_devanagari(raw):
        if any(p in raw for p in FUTURE_SOFT_DEVANAGARI):
            return "soft_no"

    # ── Step 5: Hard No (Strict rejections) ──────────────────────────────
    if contains_devanagari(raw):
        if any(p in raw for p in HARD_NO_DEVANAGARI):
            return "no"
        for pattern in HARD_NO_DEVANAGARI_PATTERNS:
            if re.search(pattern, raw):
                return "no"
        if "नहीं" in raw or "नही" in raw:
            return "no"
    
    if HARD_NO_REGEX.search(norm):
        return "no"
    if re.fullmatch(r"(no|nope|nahi|nahin|nah|mat|na|never)", norm.strip()):
        return "no"

    # ── Step 6: YES — interested ──────────────────────────────────────────
    if YES_REGEX.search(norm):
        return "yes"
    if contains_devanagari(raw):
        if any(p in raw for p in YES_DEVANAGARI):
            return "yes"

    # ── Step 7: Pure Budget Signal (if not caught as correction) ─────────
    if has_budget_signal(raw):
        return "has_budget"

    # ── Step 8: Pure Greeting — ONLY if nothing else matched ─────────────
    # (prevents "no" or "yes" alone from being "greeting_only")
    if is_greeting_only(raw):
        return "greeting_only"

    return "unclear"


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE TEST SUITE
# Run: python intent_engine.py --test
# All 60 cases should pass — if any fail, fix classify_intent above
# ─────────────────────────────────────────────────────────────────────────────
INTENT_TEST_SUITE = [
    # ── Hard no ──────────────────────────────────────────────────────────
    ("नहीं मैं नहीं कर रहा हूं",             "no",         "Hindi: I'm not doing it"),
    ("मेरे को जाना ही नहीं है",              "no",         "Hindi: I don't want to go at all"),
    ("मुझे कहीं नहीं जाना है",               "no",         "Hindi: I don't want to go anywhere"),
    ("कहीं नहीं जाना",                       "no",         "Hindi: Not going anywhere"),
    ("नहीं चाहिए",                           "no",         "Hindi: Don't want it"),
    ("बिल्कुल नहीं",                         "no",         "Hindi: Absolutely not"),
    ("छोड़ो",                                "no",         "Hindi: Leave it"),
    ("रहने दो",                              "no",         "Hindi: Let it be"),
    ("nahi nahi bilkul nahi",               "no",         "Hinglish: Absolute no"),
    ("not interested",                      "no",         "English: Not interested"),
    ("don't call me",                       "no",         "English: Don't call"),
    ("no thanks",                           "no",         "English: No thanks"),
    ("nahi chahiye",                        "no",         "Hinglish: Don't want"),
    ("nahi jaana",                          "no",         "Hinglish: Don't want to go"),
    ("remove karo",                         "no",         "Hinglish: Remove me"),
    ("no",                                  "no",         "English: Single no"),
    ("nahi",                                "no",         "Hinglish: Single nahi"),
    ("nope",                                "no",         "English: Nope"),

    # ── Soft no ──────────────────────────────────────────────────────────
    ("sochenge",                            "soft_no",    "Hinglish: Will think"),
    ("baad mein dekhenge",                  "soft_no",    "Hinglish: Will see later"),
    ("wife se poochna hai",                 "soft_no",    "Hinglish: Need to ask wife"),
    ("not sure abhi",                       "soft_no",    "Hinglish: Not sure yet"),
    ("abhi nahi par baad mein jaunga",      "soft_no",    "Hinglish: Not now but later"),
    ("thoda time chahiye",                  "soft_no",    "Hinglish: Need some time"),
    ("सोचेंगे",                             "soft_no",    "Hindi: Will think"),
    ("बाद में जाऊंगा",                      "soft_no",    "Hindi: Will go later"),
    ("family se poochna hai",               "soft_no",    "Hinglish: Need to ask family"),
    ("kal batata hoon",                     "soft_no",    "Hinglish: Will tell tomorrow"),

    # ── Busy ─────────────────────────────────────────────────────────────
    ("busy hoon abhi",                      "busy",       "Hinglish: Currently busy"),
    ("meeting mein hoon",                   "busy",       "Hinglish: In meeting"),
    ("gari chala raha hoon",                "busy",       "Hinglish: Driving"),
    ("call back karo",                      "busy",       "Hinglish: Call back"),
    ("not a good time",                     "busy",       "English: Bad time"),
    ("driving",                             "busy",       "English: Driving"),

    # ── Yes ───────────────────────────────────────────────────────────────
    ("haan",                                "yes",        "Hinglish: Yes"),
    ("yes",                                 "yes",        "English: Yes"),
    ("bilkul",                              "yes",        "Hinglish: Absolutely"),
    ("jaana hai",                           "yes",        "Hinglish: Want to go"),
    ("हाँ",                                 "yes",        "Hindi: Yes"),
    ("हाँ बताइए",                           "yes",        "Hindi: Yes tell me"),
    ("trip plan kar raha hoon",             "yes",        "Hinglish: Planning trip"),
    ("interested hoon",                     "yes",        "Hinglish: Interested"),
    ("i want to go",                        "yes",        "English: I want to go"),
    ("theek hai bataiye",                   "yes",        "Hinglish: Ok tell me"),

    # ── Budget override (no + number = budget correction) ────────────────
    ("नहीं 5 लाख",                          "has_budget", "Hindi: No 5 lakh = budget correction"),
    ("nahi 50000",                          "has_budget", "Hinglish: No 50000 = correction"),
    ("nahi 1 lakh budget hai",              "has_budget", "Hinglish: budget = 1 lakh"),
    ("50 hazaar hai budget",                "has_budget", "Hinglish: budget 50k"),
    ("60000 hai",                           "has_budget", "Number = budget"),
    ("1 lakh tak spend kar sakta hoon",     "has_budget", "Hinglish: can spend 1 lakh"),

    # ── Filler ────────────────────────────────────────────────────────────
    ("umm",                                 "filler",     "Filler: umm"),
    ("hmm",                                 "filler",     "Filler: hmm"),
    ("uh",                                  "filler",     "Filler: uh"),

    # ── Greeting ─────────────────────────────────────────────────────────
    ("hello",                               "greeting_only", "Greeting: hello"),
    ("हेलो",                                "greeting_only", "Greeting: हेलो"),
    ("namaste",                             "greeting_only", "Greeting: namaste"),
    ("hi",                                  "greeting_only", "Greeting: hi"),

    # ── Unclear ───────────────────────────────────────────────────────────
    ("कोई ऐसा बोल",                         "unclear",    "Unclear: confused"),
    ("pata nahi kuch",                      "unclear",    "Unclear: don't know"),
    ("acha",                                "unclear",    "Unclear: ambiguous"),
]


def run_tests() -> tuple[int, int]:
    """Run all test cases. Returns (passed, total)."""
    passed = 0
    failed = []

    print(f"\nRunning {len(INTENT_TEST_SUITE)} intent classification tests...\n")

    for text, expected, description in INTENT_TEST_SUITE:
        result = classify_intent(text)
        if result == expected:
            passed += 1
        else:
            failed.append((text, expected, result, description))

    print(f"Results: {passed}/{len(INTENT_TEST_SUITE)} passed\n")

    if failed:
        print(f"FAILED ({len(failed)} cases):")
        for text, expected, got, desc in failed:
            print(f"  FAIL — {desc}")
            print(f"         Input   : {repr(text)}")
            print(f"         Expected: {expected}")
            print(f"         Got     : {got}")
            print()
    else:
        print("All tests passed!")

    return passed, len(INTENT_TEST_SUITE)


# ─────────────────────────────────────────────────────────────────────────────
# FAST TRAINING — System Prompt Optimizer
# The fastest way to improve Simran's responses without any GPU training
# Tests multiple prompt variants and keeps the best one
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_VARIANTS = [
    {
        "name": "warm_hindi",
        "description": "Warm, simple Hindi — like talking to a friend",
        "system": """Aap Simran hain. Trip in Minutes (tripinminutes.com) mein kaam karti hain.
Aapko 10 saal ka travel experience hai.
Aap bilkul seedhi, simple baat karti hain — jaise ek dost baat karta hai.

KAAM: Destination aur budget pata karo, phir warmly bye karo.
LANGUAGE: Customer jis boli mein bole, usi mein jawab do.
RESPECT: Hamesha "aap" — kabhi "tum" nahi.
REPLY: Max 2 choti lines. Ek sawaal. Bus.
NO "ji" repeatedly. No robot phrases.""",
    },
    {
        "name": "direct_telecaller",
        "description": "Direct, experienced telecaller style",
        "system": """You are Simran, a senior travel telecaller at Trip in Minutes (tripinminutes.com).
10 years experience. You are warm, direct, and efficient.

GOAL: Get destination + budget in 30 seconds. Then close warmly.
LANGUAGE: Always match customer — Hindi reply for Hindi, English for English, Hinglish for Hinglish.
STYLE: Real Indian woman on a phone call. Simple daily words. Never robotic.
FORMAT: Max 2 short sentences. One question. No "ji" repeatedly. No "Certainly" or "Absolutely".""",
    },
    {
        "name": "conversational_hinglish",
        "description": "Natural Hinglish — most common Indian customer style",
        "system": """Tum Simran ho. Trip in Minutes (tripinminutes.com) ki telecaller.
10 saal ka experience hai travel mein.

Simple kaam: pehle destination pata karo, phir budget, phir bye.
Jo customer ki language hai, waisi baat karo.
"Aap" bolna — "tum" mat bolna.
2 se zyada lines mat bolna. Ek sawal karo, bas.
Seedha, warm, natural — jaise WhatsApp pe baat karte hain.""",
    },
    {
        "name": "empathetic_listener",
        "description": "Empathetic, listens first, then guides",
        "system": """Aap Simran hain — Trip in Minutes ki experienced travel advisor.
Aap pehle customer ki baat sunti hain, phir helpful reply deti hain.

Kaam: destination + budget collect karo warmly.
Language: customer ki boli mein hi bolna — Hindi/English/Hinglish match karo.
Tone: samajhdaar dost ki tarah — pushful nahi, helpful.
Reply format: 1-2 short sentences max. 1 question. Simple words.""",
    },
]

# Test conversations to evaluate each prompt variant
EVAL_CONVERSATIONS = [
    {
        "customer": "haan boliye",
        "context": "stage:greeting lang:hinglish",
        "ideal_contains": ["trip", "kahan", "plan", "jaana"],
        "ideal_not_contains": ["certainly", "absolutely", "ji ji"],
    },
    {
        "customer": "thailand jaana hai",
        "context": "stage:got_interest lang:hinglish dest:none",
        "ideal_contains": ["budget", "kitna", "per person"],
        "ideal_not_contains": ["destination", "kahan jaana", "tell me"],
    },
    {
        "customer": "Is it expensive?",
        "context": "stage:got_destination lang:english dest:Bali",
        "ideal_contains": ["budget", "range", "options"],
        "ideal_not_contains": ["nahi", "koi"],
    },
    {
        "customer": "मुझे गोवा जाना है",
        "context": "stage:greeting lang:hindi",
        "ideal_contains": ["budget", "kitna", "goa"],
        "ideal_not_contains": ["destination", "kahan"],
    },
    {
        "customer": "sochenge baad mein",
        "context": "stage:got_interest lang:hinglish",
        "ideal_contains": ["tripinminutes", "options", "koi"],
        "ideal_not_contains": ["kahan jaana", "destination"],
    },
]


def score_response(response: str, eval_case: dict) -> float:
    """Score a response 0-10 based on ideal criteria."""
    score = 5.0
    r     = response.lower()

    # Check ideal contains
    contains_hits = sum(1 for w in eval_case["ideal_contains"] if w.lower() in r)
    score += (contains_hits / max(len(eval_case["ideal_contains"]), 1)) * 3

    # Check ideal not contains (penalize)
    bad_hits = sum(1 for w in eval_case["ideal_not_contains"] if w.lower() in r)
    score -= bad_hits * 1.5

    # Length check — phone call, keep it short
    words = len(response.split())
    if words <= 20:  score += 1.0
    elif words <= 35: score += 0.5
    else:            score -= 1.0

    # Question mark check — should ask one question
    if response.count("?") == 1: score += 0.5
    if response.count("?") > 1:  score -= 1.0

    return max(0.0, min(10.0, score))


def optimize_system_prompt(n_eval: int = 5) -> str:
    """
    Test all prompt variants against eval conversations.
    Returns the best performing prompt variant name.
    Saves results to prompt_scores.json.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        print("GROQ_API_KEY not set. Cannot run optimization.")
        return "warm_hindi"

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_key,
    )

    print(f"\nOptimizing system prompt across {len(PROMPT_VARIANTS)} variants...")
    print(f"Testing {len(EVAL_CONVERSATIONS)} conversations each\n")

    results = {}

    for variant in PROMPT_VARIANTS:
        print(f"Testing variant: {variant['name']} — {variant['description']}")
        scores = []

        for eval_case in EVAL_CONVERSATIONS[:n_eval]:
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": variant["system"]},
                        {"role": "user",   "content":
                         f"[{eval_case['context']}]\nCustomer: {eval_case['customer']}"},
                    ],
                    max_tokens=80,
                    temperature=0.3,
                )
                reply = resp.choices[0].message.content.strip()
                score = score_response(reply, eval_case)
                scores.append(score)
                print(f"  Q: {eval_case['customer'][:40]}")
                print(f"  A: {reply[:80]}")
                print(f"  Score: {score:.1f}/10\n")
                time.sleep(1)

            except Exception as e:
                print(f"  Error: {e}")
                scores.append(5.0)

        avg = sum(scores) / len(scores) if scores else 0
        results[variant["name"]] = {
            "avg_score": avg,
            "scores":    scores,
            "prompt":    variant["system"],
        }
        print(f"  → Average score: {avg:.1f}/10\n")

    # Save results
    with open("prompt_scores.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Find best
    best_name = max(results, key=lambda k: results[k]["avg_score"])
    best_score = results[best_name]["avg_score"]

    print("=" * 50)
    print("PROMPT OPTIMIZATION RESULTS")
    print("=" * 50)
    for name, data in sorted(results.items(), key=lambda x: -x[1]["avg_score"]):
        bar = "█" * int(data["avg_score"])
        print(f"  {name:<25} {data['avg_score']:.1f}/10  {bar}")
    print()
    print(f"Best variant: {best_name} ({best_score:.1f}/10)")
    print(f"Results saved to: prompt_scores.json")
    print()
    print("To use the best prompt, copy this into SIMRAN_SYSTEM in main.py:")
    print("-" * 50)
    print(results[best_name]["prompt"])
    print("-" * 50)

    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# FAST FEW-SHOT INJECTOR
# The single fastest way to improve response quality — no training needed
# Injects best examples directly into every LLM call
# ─────────────────────────────────────────────────────────────────────────────
GOLD_FEW_SHOT = {
    # Format: (customer_says, simran_should_reply)
    # These are injected as examples before every LLM call
    "hinglish_greeting": [
        ("haan boliye",
         "Hello! Kaise hain aap? Koi trip plan chal rahi hai?"),
        ("hello",
         "Hello! Bataiye, koi ghumne ka plan hai aajkal?"),
    ],
    "hinglish_destination": [
        ("thailand jaana hai",
         "Thailand bahut sundar jagah hai! Per person roughly kitna budget soch rahe hain?"),
        ("goa jaana chahte hain",
         "Goa! Wahan bahut maza aata hai. Budget roughly kitna hai aapka?"),
        ("bali ya dubai mein se kaunsa better hai",
         "Dono acche hain — Bali zyada peaceful hai, Dubai thoda zyada exciting. Aap kya prefer karte hain?"),
    ],
    "hinglish_budget": [
        ("50 hazaar hai budget",
         "Bahut accha! Hamare consultant aapko best options ke saath jald call karenge. Aapka time dene ka shukriya!"),
        ("1 lakh tak spend kar sakta hoon",
         "Perfect! Hamare travel expert aapko Dubai ke kuch acche packages ke saath jald call karenge."),
    ],
    "hindi_greeting": [
        ("हेलो",
         "Hello! Kaise hain aap? Koi trip plan chal rahi hai?"),
        ("नमस्ते",
         "Namaste! Bataiye, koi ghumne ka plan hai?"),
    ],
    "hindi_destination": [
        ("थाईलैंड जाना है",
         "Thailand bahut sundar jagah hai! Budget roughly kitna soch rahe hain per person?"),
        ("गोवा जाना है",
         "Goa! Wahan bahut maza aata hai. Per person kitna budget hai roughly?"),
    ],
    "english_greeting": [
        ("hello",
         "Hello! How are you? Are you planning any trips these days?"),
        ("hi",
         "Hi! Any travel plans going on?"),
    ],
    "english_destination": [
        ("I want to go to Bali",
         "Bali is a beautiful choice! What's your rough budget per person?"),
        ("thinking of going to Dubai",
         "Dubai is great! What's your approximate budget for the trip?"),
    ],
    "soft_no": [
        ("sochenge",
         "Koi baat nahi, sochiye! Jab ready hon, tripinminutes.com pe best deals milti hain."),
        ("baad mein dekhenge",
         "Bilkul! Koi jaldi nahi. Koi bhi sawaal ho toh tripinminutes.com pe aa sakte hain."),
    ],
    "question": [
        ("visa chahiye kya thailand ke liye",
         "Nahi, Indians ko Thailand mein visa on arrival milta hai, bilkul easy. Kab jaana soch rahe hain?"),
        ("is it safe",
         "Yes, very safe for Indian tourists. When are you planning to travel?"),
    ],
}


def get_few_shot_messages(language: str, stage: str) -> list[dict]:
    """
    Get few-shot examples as message history.
    Inject these BEFORE the actual user message in every LLM call.
    
    Usage in main.py:
        few_shots = get_few_shot_messages(self.customer_language, self.stage)
        messages = [
            {"role": "system", "content": SIMRAN_SYSTEM},
            *few_shots,                    # inject examples here
            *self.history[-4:],            # then real conversation
            {"role": "user", "content": customer_text},
        ]
    """
    key = f"{language}_{stage_to_key(stage)}"

    # Try exact match first, then fall back to language only
    examples = (
        GOLD_FEW_SHOT.get(key) or
        GOLD_FEW_SHOT.get(f"{language}_greeting") or
        GOLD_FEW_SHOT.get("hinglish_greeting") or
        []
    )

    # Pick best 2 examples, format as message history
    messages = []
    for customer_msg, simran_reply in examples[:2]:
        messages.append({"role": "user",      "content": customer_msg})
        messages.append({"role": "assistant", "content": simran_reply})
    return messages


def stage_to_key(stage: str) -> str:
    mapping = {
        "greeting":         "greeting",
        "got_interest":     "destination",
        "got_destination":  "budget",
        "done":             "budget",
    }
    return mapping.get(stage, "greeting")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simran Intent Engine — test classifier and optimize prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  python intent_engine.py --test        Run all 60 intent tests
  python intent_engine.py --optimize    Find best system prompt (uses Groq API)
  python intent_engine.py --classify "text"   Classify a single input

Examples:
  python intent_engine.py --test
  python intent_engine.py --classify "नहीं मैं नहीं कर रहा हूं"
  python intent_engine.py --classify "sochenge baad mein"
  python intent_engine.py --optimize
        """
    )
    parser.add_argument("--test",     action="store_true", help="Run full test suite")
    parser.add_argument("--optimize", action="store_true", help="Optimize system prompt")
    parser.add_argument("--classify", type=str, help="Classify a single input text")
    args = parser.parse_args()

    if args.test:
        passed, total = run_tests()
        exit(0 if passed == total else 1)

    elif args.optimize:
        best = optimize_system_prompt()
        print(f"\nBest prompt: {best}")

    elif args.classify:
        result = classify_intent(args.classify)
        print(f"Input : {args.classify}")
        print(f"Intent: {result}")

    else:
        # Run tests by default
        run_tests()
