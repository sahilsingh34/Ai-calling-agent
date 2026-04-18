"""
prompt_engine.py — Approach 1 + Approach 2 for Simran
=======================================================
Approach 1: Few-shot injection  — inject perfect examples into every LLM call
Approach 2: System prompt optimizer — test and find the best prompt automatically

Both work TODAY with zero training and zero GPU.

HOW TO USE:
-----------
Step 1 — Run the optimizer to find the best system prompt:
    python prompt_engine.py --optimize

Step 2 — Test few-shot injection manually:
    python prompt_engine.py --test-fewshot

Step 3 — Copy the 3-line integration into main.py (shown at bottom of this file)

Run anytime to improve:
    python prompt_engine.py --optimize     # re-run after adding new examples
    python prompt_engine.py --add-example  # add a new perfect example
"""

import json
import os
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load env variables (API keys, etc.)
load_dotenv()

GROQ_CLIENT = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", ""),
)
GROQ_MODEL = "llama-3.1-8b-instant"  # Use 8B for benchmarking to avoid TPD limits
SCORES_FILE = Path("prompt_scores.json")
DB_PATH     = Path("simran_training.db")


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 1 — FEW-SHOT INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Perfect example conversations — hand-crafted gold standard
# Format: (customer_says, simran_perfect_reply)
# These get injected into every LLM call as demonstrations
# The model sees these and COPIES the style automatically

FEW_SHOT_BANK = {

    # ── Hinglish examples (most common customer type) ─────────────────────
    "hinglish_greeting": [
        ("haan boliye",
         "Hello! Kaise hain aap? Koi trip plan chal rahi hai?"),
        ("hello bhai",
         "Hello! Bataiye, koi ghumne ka plan hai aajkal?"),
        ("haan baat karein",
         "Haan! Koi trip soch rahe hain aap — domestic ya international?"),
    ],

    "hinglish_destination": [
        ("thailand jaana hai",
         "Thailand bahut sundar jagah hai! Per person roughly kitna budget soch rahe hain?"),
        ("goa jaana chahte hain family ke saath",
         "Goa! Wahan bahut maza aata hai. Budget roughly kitna hai aapka — per person?"),
        ("dubai ya bali mein se kaunsa better rahega",
         "Dono acche hain — Dubai thoda zyada exciting, Bali zyada peaceful. Aap kya prefer karte hain?"),
        ("kashmir jaana hai is baar",
         "Kashmir sach mein bahut sundar jagah hai! Budget roughly kitna soch rahe hain per person?"),
        ("europe trip plan kar raha hoon",
         "Europe! Bahut accha soch rahe hain. Kaun sa country mainly — Paris, Switzerland, ya aur koi?"),
    ],

    "hinglish_budget": [
        ("50 hazaar hai budget",
         "Bahut accha! Hamare consultant aapko best options ke saath jald call karenge. Aapka time dene ka shukriya!"),
        ("1 lakh tak spend kar sakta hoon",
         "Perfect! Hamare travel expert aapko Dubai ke kuch acche packages ke saath jald call karenge."),
        ("budget 30-40 hazaar ke beech hai",
         "Bahut accha! Hamare consultant aapko Goa ke best packages ke saath jald call karenge. Shukriya!"),
    ],

    "hinglish_question": [
        ("visa chahiye kya thailand ke liye",
         "Nahi, Indians ko Thailand mein visa on arrival milta hai — bilkul easy. Kab jaana soch rahe hain roughly?"),
        ("kitna expensive hai maldives",
         "Per person roughly 80 hazaar se 1.5 lakh tak hota hai. Aapka budget approximately kitna hai?"),
        ("yeh company kaisi hai",
         "Hum 10 saal se thousands of customers ko trip karwa rahe hain — trusted naam hai. Aap kahan jaana chahte hain?"),
        ("best time kab hai goa ke liye",
         "October se March sabse accha time hai — weather perfect rehta hai. Budget roughly kitna soch rahe hain?"),
        ("safe hai kya thailand travel ke liye",
         "Haan bilkul safe hai — bahut Indians jaate hain wahan. Budget roughly kitna hai per person?"),
    ],

    "hinglish_soft_no": [
        ("sochenge",
         "Koi baat nahi, bilkul sochiye! Jab bhi ready hon — tripinminutes.com pe best deals milti hain anytime."),
        ("wife se poochna hai pehle",
         "Bilkul! Ghar mein discuss karein. Koi bhi sawaal ho toh tripinminutes.com pe bhi dekh sakte hain."),
        ("abhi nahi par baad mein jaana hai",
         "Koi baat nahi! Jab plan karein — hum available hain. tripinminutes.com pe bhi dekh sakte hain."),
    ],

    # ── Hindi examples (Devanagari speakers) ──────────────────────────────
    "hindi_greeting": [
        ("हेलो",
         "Hello! Kaise hain aap? Koi trip plan chal rahi hai?"),
        ("नमस्ते",
         "Namaste! Bataiye, koi ghumne ka plan hai aajkal?"),
        ("हाँ बोलिए",
         "Haan! Koi trip soch rahe hain — kahan jaana hai?"),
    ],

    "hindi_destination": [
        ("थाईलैंड जाना है",
         "Thailand bahut sundar jagah hai! Per person roughly kitna budget soch rahe hain?"),
        ("गोवा जाने का सोच रहे हैं",
         "Goa! Wahan bahut maza aata hai. Budget roughly kitna hai aapka?"),
        ("कश्मीर जाना है",
         "Kashmir sach mein bahut sundar jagah hai — jannat hai India ka! Budget roughly kitna soch rahe hain?"),
        ("मालदीव जाना है",
         "Maldives! Sapne wali jagah hai. Budget roughly kitna hai aapka per person?"),
    ],

    "hindi_budget": [
        ("50 हज़ार है",
         "Bahut accha! Hamare consultant aapko best options ke saath jald call karenge. Shukriya!"),
        ("1 लाख तक खर्च कर सकते हैं",
         "Perfect! Hamare travel expert aapko jald call karenge acche packages ke saath."),
    ],

    "hindi_question": [
        ("वीजा चाहिए क्या",
         "Nahi, Indians ko wahan visa on arrival milta hai, koi tension nahi. Kab jaana soch rahe hain?"),
        ("कितना महंगा है",
         "30 hazaar se lekar zyada tak options hote hain. Aapka roughly kitna budget hai?"),
    ],

    # ── English examples ────────────────────────────────────────────────────
    "english_greeting": [
        ("hello",
         "Hello! How are you? Are you planning any trips these days?"),
        ("hi there",
         "Hi! Any travel plans going on? Where are you thinking of going?"),
        ("yes please",
         "Great! Where are you planning to travel — any destination in mind?"),
    ],

    "english_destination": [
        ("I want to go to Bali",
         "Bali is a beautiful choice! What's your rough budget per person?"),
        ("thinking of Dubai",
         "Dubai is great — shopping, adventure, everything! What's your approximate budget?"),
        ("we want to do a Europe trip",
         "Europe! Amazing choice. Which country mainly — France, Switzerland, or somewhere else?"),
        ("planning a trip to Thailand",
         "Thailand is wonderful! Great beaches and food. What's your rough budget per person?"),
    ],

    "english_budget": [
        ("around 60000",
         "Perfect! Our travel consultant will call you shortly with the best options. Thank you!"),
        ("budget is about 1 lakh per person",
         "Excellent! Our travel expert will call you soon with great packages. Have a great day!"),
    ],

    "english_question": [
        ("do I need a visa for Thailand",
         "No visa needed — Indians get visa on arrival in Thailand, very easy. When are you planning to go?"),
        ("is it safe to travel there",
         "Yes, very safe for Indian tourists — thousands visit every year. What's your rough budget?"),
        ("how much does it cost",
         "Packages start from around 40,000 per person depending on hotel and dates. What's your budget roughly?"),
    ],
}


def get_few_shot_messages(language: str, stage: str, n: int = 2) -> list[dict]:
    """
    Get n perfect examples for this language + stage combination.
    Returns as message history — ready to inject into LLM call.

    Usage in main.py:
        few_shots = get_few_shot_messages(self.customer_language, self.stage)
        messages = [
            {"role": "system", "content": SIMRAN_SYSTEM},
            *few_shots,                       # ← inject here
            *self.history[-HISTORY_WINDOW:],
            {"role": "user", "content": ...}
        ]
    """
    # Map stage to key
    stage_key = {
        "greeting":        "greeting",
        "got_interest":    "destination",
        "got_destination": "budget",
        "done":            "budget",
    }.get(stage, "greeting")

    key = f"{language}_{stage_key}"

    # Try exact match, then language-only fallback
    examples = (
        FEW_SHOT_BANK.get(key) or
        FEW_SHOT_BANK.get(f"{language}_greeting") or
        FEW_SHOT_BANK.get("hinglish_greeting") or
        []
    )

    # Also add a question example if we have one (always useful)
    q_examples = FEW_SHOT_BANK.get(f"{language}_question") or \
                 FEW_SHOT_BANK.get("hinglish_question") or []

    # Build message list from examples
    messages = []
    used = set()

    # Add stage-specific examples first
    for customer_msg, simran_reply in examples[:n]:
        if customer_msg not in used:
            messages.append({"role": "user",      "content": customer_msg})
            messages.append({"role": "assistant", "content": simran_reply})
            used.add(customer_msg)

    # Add one question example
    for customer_msg, simran_reply in q_examples[:1]:
        if customer_msg not in used:
            messages.append({"role": "user",      "content": customer_msg})
            messages.append({"role": "assistant", "content": simran_reply})
            used.add(customer_msg)

    return messages


def build_few_shot_header(language: str, stage: str) -> str:
    """
    Alternative: build few-shot examples as a text block for the system prompt.
    Use this if you prefer to put examples in the system prompt instead of messages.
    """
    stage_key = {
        "greeting":        "greeting",
        "got_interest":    "destination",
        "got_destination": "budget",
    }.get(stage, "greeting")

    key      = f"{language}_{stage_key}"
    examples = FEW_SHOT_BANK.get(key) or FEW_SHOT_BANK.get("hinglish_greeting") or []

    if not examples:
        return ""

    lines = ["\n\nPERFECT EXAMPLES — follow this exact style:"]
    for customer_msg, simran_reply in examples[:3]:
        lines.append(f"\nCustomer: {customer_msg}")
        lines.append(f"Simran:   {simran_reply}")
    return "\n".join(lines)


def add_example(language: str, stage: str, customer_msg: str, simran_reply: str):
    """
    Add a new perfect example to the bank at runtime.
    Call this when you find a great response in real calls.
    """
    stage_key = {
        "greeting":        "greeting",
        "got_interest":    "destination",
        "got_destination": "budget",
    }.get(stage, "greeting")

    key = f"{language}_{stage_key}"
    if key not in FEW_SHOT_BANK:
        FEW_SHOT_BANK[key] = []

    # Avoid duplicates
    existing = [ex[0] for ex in FEW_SHOT_BANK[key]]
    if customer_msg not in existing:
        FEW_SHOT_BANK[key].append((customer_msg, simran_reply))
        print(f"Added to '{key}': {customer_msg[:40]} → {simran_reply[:40]}")
    else:
        print("Example already exists.")


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 2 — SYSTEM PROMPT OPTIMIZER
# Tests multiple prompt variants, scores them, keeps the best
# ═══════════════════════════════════════════════════════════════════════════════

# 6 different prompt styles — optimizer finds the best one for YOUR customers
PROMPT_VARIANTS = {

    "warm_hinglish": {
        "description": "Warm Hinglish — like talking to a dost",
        "prompt": """Aap Simran hain. Trip in Minutes (tripinminutes.com) mein kaam karti hain.
10 saal ka travel experience hai. Phone pe bilkul natural baat karti hain — jaise ek dost.

SIRF YEH KAAM: destination pata karo → budget pata karo → warmly bye
LANGUAGE: customer ki boli mein hi bolein — Hindi/English/Hinglish match karein
RESPECT: hamesha "aap", kabhi "tum" nahi
FORMAT: max 2 choti lines, ek sawal, warm tone
KABHI NAHI: "Certainly", "Absolutely", "ji ji ji", same cheez dobara""",
    },

    "direct_telecaller": {
        "description": "Direct experienced telecaller — sharp and warm",
        "prompt": """You are Simran, senior travel telecaller at Trip in Minutes (tripinminutes.com).
10 years experience. Sharp, warm, efficient.

YOUR ONLY JOB: Get destination + budget in 30 seconds. Close warmly.
LANGUAGE RULE: Always match customer — Hindi→Hindi, English→English, Hinglish→Hinglish.
STYLE: Real Indian woman on phone. Simple words. Never robotic.
FORMAT: Max 2 short sentences. One question. No "ji" repeatedly. No "Certainly" or "Absolutely".
NEVER ask something customer already told you.""",
    },

    "conversational_natural": {
        "description": "Most natural — WhatsApp voice note style",
        "prompt": """Simran ho tum — Trip in Minutes (tripinminutes.com) ki telecaller.
10 saal ka experience. Seedha, warm, real.

Kaam: destination pata karo, budget pata karo, bye.
Jo language customer use kare — wohi use karo.
"Aap" bolna. Max 2 lines. Ek sawaal.
Jaise WhatsApp pe koi dost baat kare — waise bolna.
"Certainly/Absolutely/Of course" — kabhi mat bolna.""",
    },

    "empathetic_expert": {
        "description": "Empathetic travel expert — listens then guides",
        "prompt": """Aap Simran hain — Trip in Minutes (tripinminutes.com) ki travel expert.
Customer ki baat sunti hain pehle. Phir helpful, warm reply deti hain.

Goal: destination + budget collect karo, phir warmly close karo.
Language: customer ki boli — Hindi/Hinglish/English — match karein.
Tone: samajhdaar dost — helpful, never pushy.
Reply: 1-2 short sentences. 1 question. Simple everyday words.
Never repeat what customer already said.""",
    },

    "hindi_first": {
        "description": "Hindi-first — best for Hindi-speaking customers",
        "prompt": """आप Simran हैं। Trip in Minutes (tripinminutes.com) में काम करती हैं।
10 साल का travel experience। Phone पे real Indian aurat की tarah baat karti hain।

KAAM: destination + budget pata karo, warmly bye karo।
LANGUAGE: Customer जिस भाषा में बोले, उसी में जवाब दें।
RESPECT: हमेशा "आप", कभी "तुम" नहीं।
FORMAT: Max 2 choti lines। Ek sawaal। Simple words।
KABHI NAHI: "Certainly", "ji ji ji", dobara wohi poochho jo bata diya।""",
    },

    "ultra_brief": {
        "description": "Ultra brief — fastest responses, minimum words",
        "prompt": """Simran — Trip in Minutes telecaller. 10 years travel experience.
Goal: get destination, get budget, close warmly. Nothing else.
Match customer's language every reply.
Use "aap". Max 1-2 sentences. 1 question only.
React warmly to destination. Give budget range for their city.
Never robotic. Never repeat info customer gave.""",
    },
}

# Test cases — covers all real customer types
EVAL_CASES = [
    {
        "name":    "hinglish hello",
        "lang":    "hinglish",
        "stage":   "greeting",
        "input":   "haan boliye",
        "good_if": ["trip","plan","kahan","jaana","ghumne"],
        "bad_if":  ["certainly","absolutely","ji ji","destination poochh"],
    },
    {
        "name":    "hindi hello",
        "lang":    "hindi",
        "stage":   "greeting",
        "input":   "हेलो",
        "good_if": ["trip","plan","kahan","jaana","ghumne","hello"],
        "bad_if":  ["certainly","destination","batao abhi"],
    },
    {
        "name":    "destination given",
        "lang":    "hinglish",
        "stage":   "got_interest",
        "input":   "thailand jaana hai",
        "good_if": ["budget","kitna","per person","hazaar"],
        "bad_if":  ["destination","kahan jaana","certainly","absolutely"],
    },
    {
        "name":    "budget given",
        "lang":    "hinglish",
        "stage":   "got_destination",
        "input":   "50 hazaar hai budget",
        "good_if": ["consultant","call","karenge","shukriya","thank"],
        "bad_if":  ["destination","kahan","budget kitna"],
    },
    {
        "name":    "visa question",
        "lang":    "hinglish",
        "stage":   "got_destination",
        "input":   "visa chahiye kya thailand ke liye",
        "good_if": ["visa","on arrival","nahi chahiye","easy"],
        "bad_if":  ["certainly","absolutely","i understand"],
    },
    {
        "name":    "price question",
        "lang":    "hinglish",
        "stage":   "got_interest",
        "input":   "kitna lagega goa jaane mein",
        "good_if": ["hazaar","budget","kitna","options"],
        "bad_if":  ["certainly","absolutely","great question"],
    },
    {
        "name":    "english hello",
        "lang":    "english",
        "stage":   "greeting",
        "input":   "hello",
        "good_if": ["trip","plan","travel","going"],
        "bad_if":  ["haan","nahi","kahan","certainly"],
    },
    {
        "name":    "english destination",
        "lang":    "english",
        "stage":   "got_interest",
        "input":   "I want to go to Bali",
        "good_if": ["budget","per person","rough","how much"],
        "bad_if":  ["haan","nahi","kahan","ji"],
    },
    {
        "name":    "soft no",
        "lang":    "hinglish",
        "stage":   "greeting",
        "input":   "sochenge baad mein",
        "good_if": ["tripinminutes","koi baat nahi","available","ready"],
        "bad_if":  ["destination poochh","kahan jaana","budget"],
    },
    {
        "name":    "hindi destination",
        "lang":    "hindi",
        "stage":   "got_interest",
        "input":   "थाईलैंड जाना है",
        "good_if": ["budget","kitna","per person","hazaar"],
        "bad_if":  ["destination","kahan","certainly"],
    },
]


def score_response(response: str, case: dict) -> float:
    """Score a Simran response 0-10 based on quality criteria."""
    r     = response.lower()
    score = 5.0

    # Good signals present
    good_hits = sum(1 for w in case["good_if"] if w.lower() in r)
    score += (good_hits / max(len(case["good_if"]), 1)) * 3.0

    # Bad signals penalised
    bad_hits = sum(1 for w in case["bad_if"] if w.lower() in r)
    score -= bad_hits * 1.5

    # Length — phone call needs to be short
    words = len(response.split())
    if words <= 18:   score += 1.0
    elif words <= 30: score += 0.5
    elif words > 45:  score -= 1.5

    # Exactly one question mark
    q_count = response.count("?")
    if q_count == 1:  score += 0.5
    elif q_count > 1: score -= 1.0
    elif q_count == 0: score -= 0.5

    return round(max(0.0, min(10.0, score)), 2)


def optimize_prompts(use_few_shot: bool = True) -> str:
    """
    Test all prompt variants against eval cases.
    Optionally combine with few-shot injection.
    Returns name of best variant.
    Saves results to prompt_scores.json.
    """
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set in .env — cannot run optimization.")
        return "warm_hinglish"

    mode = "with few-shot" if use_few_shot else "prompt only"
    print(f"\nRunning prompt optimization ({mode})")
    print(f"Testing {len(PROMPT_VARIANTS)} variants × {len(EVAL_CASES)} cases\n")

    all_results = {}

    for variant_name, variant in PROMPT_VARIANTS.items():
        print(f"Testing: {variant_name} — {variant['description']}")
        scores = []

        for case in EVAL_CASES:
            # Build messages
            messages = [{"role": "system", "content": variant["prompt"]}]

            # Approach 1 + 2 combined — inject few-shot examples
            if use_few_shot:
                few_shots = get_few_shot_messages(case["lang"], case["stage"])
                messages.extend(few_shots)

            messages.append({
                "role":    "user",
                "content": f"[lang:{case['lang']} stage:{case['stage']}]\n{case['input']}",
            })

            try:
                resp = GROQ_CLIENT.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    max_tokens=80,
                    temperature=0.3,
                )
                reply = resp.choices[0].message.content.strip()
                score = score_response(reply, case)
                scores.append(score)

                status = "GOOD" if score >= 7 else ("OK" if score >= 5 else "POOR")
                print(f"  [{status} {score:.1f}] {case['name']}: {reply[:60]}...")
                time.sleep(0.5)

            except Exception as e:
                print(f"  [ERROR] {case['name']}: {e}")
                scores.append(5.0)
                time.sleep(2)

        avg = round(sum(scores) / len(scores), 2) if scores else 0.0
        all_results[variant_name] = {
            "avg_score":   avg,
            "scores":      scores,
            "description": variant["description"],
            "prompt":      variant["prompt"],
        }
        print(f"  → Average: {avg:.1f}/10\n")

    # Sort by score
    ranked = sorted(all_results.items(), key=lambda x: -x[1]["avg_score"])

    # Save
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print results
    print("=" * 60)
    print("  OPTIMIZATION RESULTS")
    print("=" * 60)
    for name, data in ranked:
        bar  = "█" * int(data["avg_score"])
        mark = " ← BEST" if name == ranked[0][0] else ""
        print(f"  {name:<25} {data['avg_score']:.1f}/10  {bar}{mark}")
    print()

    best_name   = ranked[0][0]
    best_prompt = ranked[0][1]["prompt"]
    best_score  = ranked[0][1]["avg_score"]

    print(f"Winner: {best_name} ({best_score:.1f}/10)")
    print(f"Saved to: {SCORES_FILE}\n")
    print("Copy this prompt into SIMRAN_SYSTEM in main.py:")
    print("-" * 60)
    print(best_prompt)
    print("-" * 60)
    print()

    # Save best prompt separately for easy copy
    with open("best_prompt.txt", "w", encoding="utf-8") as f:
        f.write(f"# Best prompt: {best_name} ({best_score:.1f}/10)\n\n")
        f.write(f'SIMRAN_SYSTEM = """{best_prompt}"""\n')
    print("Also saved to: best_prompt.txt")

    return best_name


def test_few_shot_live():
    """
    Interactive test — type a customer message, see how few-shot injection improves response.
    Compares: no few-shot vs with few-shot side by side.
    """
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set.")
        return

    print("\nFew-shot injection live test")
    print("Type a customer message to see the difference.\n")

    # Use best prompt if available, else default
    if SCORES_FILE.exists():
        with open(SCORES_FILE) as f:
            saved = json.load(f)
        best = max(saved.items(), key=lambda x: x[1]["avg_score"])
        prompt = best[1]["prompt"]
        print(f"Using best prompt: {best[0]} ({best[1]['avg_score']:.1f}/10)\n")
    else:
        prompt = PROMPT_VARIANTS["warm_hinglish"]["prompt"]
        print("Using warm_hinglish prompt (run --optimize first for best results)\n")

    while True:
        try:
            lang  = input("Language (hinglish/hindi/english) [hinglish]: ").strip() or "hinglish"
            stage = input("Stage (greeting/got_interest/got_destination) [got_interest]: ").strip() or "got_interest"
            text  = input("Customer says: ").strip()
            if not text: break

            print()

            # ── Without few-shot ───────────────────────────────────────
            resp_plain = GROQ_CLIENT.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": f"[lang:{lang} stage:{stage}]\n{text}"},
                ],
                max_tokens=80,
                temperature=0.3,
            )
            plain = resp_plain.choices[0].message.content.strip()

            # ── With few-shot ──────────────────────────────────────────
            few_shots = get_few_shot_messages(lang, stage, n=2)
            resp_fewshot = GROQ_CLIENT.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    *few_shots,
                    {"role": "user",   "content": f"[lang:{lang} stage:{stage}]\n{text}"},
                ],
                max_tokens=80,
                temperature=0.3,
            )
            with_fewshot = resp_fewshot.choices[0].message.content.strip()

            print(f"Without few-shot : {plain}")
            print(f"With few-shot    : {with_fewshot}")
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN.PY INTEGRATION — copy these 3 changes into main.py
# ═══════════════════════════════════════════════════════════════════════════════

INTEGRATION_CODE = '''
# ════════════════════════════════════════════════════════
# CHANGES TO MAKE IN main.py — copy exactly as shown
# ════════════════════════════════════════════════════════

# ── CHANGE 1 — Add import at top of main.py ──────────────
# Add after "import config":
from prompt_engine import get_few_shot_messages

# ── CHANGE 2 — In _answer_question(), update messages ────
# FIND this block:
    messages = [
        {"role": "system", "content": SIMRAN_SYSTEM},
        *self.history[-HISTORY_WINDOW:],
        {"role": "user", "content": ...}

# REPLACE with:
    few_shots = get_few_shot_messages(self.customer_language, self.stage)
    messages = [
        {"role": "system", "content": SIMRAN_SYSTEM},
        *few_shots,                        # ← few-shot examples
        *self.history[-HISTORY_WINDOW:],   # ← real conversation
        {"role": "user", "content": ...}

# ── CHANGE 3 — In _llm_nudge(), same change ──────────────
# FIND:
    messages = [
        {"role": "system", "content": SIMRAN_SYSTEM},
        *self.history[-HISTORY_WINDOW:],
        {"role": "user", "content": ...}

# REPLACE with:
    few_shots = get_few_shot_messages(self.customer_language, self.stage)
    messages = [
        {"role": "system", "content": SIMRAN_SYSTEM},
        *few_shots,
        *self.history[-HISTORY_WINDOW:],
        {"role": "user", "content": ...}

# ── CHANGE 4 — Use best prompt from optimizer ────────────
# After running: python prompt_engine.py --optimize
# Open best_prompt.txt and copy SIMRAN_SYSTEM into main.py
# ════════════════════════════════════════════════════════
'''


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simran Prompt Engine — few-shot injection + prompt optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  python prompt_engine.py --optimize          Find best system prompt (recommended first)
  python prompt_engine.py --optimize-combined Optimize prompt + few-shot together
  python prompt_engine.py --test-fewshot      Compare with vs without few-shot live
  python prompt_engine.py --integration       Print exact code to paste into main.py
  python prompt_engine.py --scores            Show previous optimization results
        """,
    )
    parser.add_argument("--optimize",          action="store_true")
    parser.add_argument("--optimize-combined", action="store_true")
    parser.add_argument("--test-fewshot",      action="store_true")
    parser.add_argument("--integration",       action="store_true")
    parser.add_argument("--scores",            action="store_true")
    args = parser.parse_args()

    if args.optimize:
        optimize_prompts(use_few_shot=False)

    elif args.optimize_combined:
        print("Testing prompt-only first...")
        optimize_prompts(use_few_shot=False)
        print("\nNow testing prompt + few-shot combined...")
        optimize_prompts(use_few_shot=True)

    elif args.test_fewshot:
        test_few_shot_live()

    elif args.integration:
        print(INTEGRATION_CODE)

    elif args.scores:
        if SCORES_FILE.exists():
            with open(SCORES_FILE) as f:
                data = json.load(f)
            print("\nPrevious optimization results:")
            for name, d in sorted(data.items(), key=lambda x: -x[1]["avg_score"]):
                bar = "█" * int(d["avg_score"])
                print(f"  {name:<25} {d['avg_score']:.1f}/10  {bar}")
        else:
            print("No results yet. Run: python prompt_engine.py --optimize")

    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python prompt_engine.py --optimize")
        print("  python prompt_engine.py --integration")
