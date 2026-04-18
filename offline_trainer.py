"""
offline_trainer.py — Simran Offline Training Infrastructure
===========================================================
Generates massive high-quality training data without needing real calls.

Methods:
1. Synthetic Generator (--synthetic) : Direct turn-by-turn generation.
2. Self-Play Simulator (--selfplay)  : Two AI agents conversing naturally.
3. Curriculum Training (--curriculum): Scaled difficulty levels.

Usage:
  python offline_trainer.py --synthetic --n 50
  python offline_trainer.py --selfplay --n 10
  python offline_trainer.py --all
  python offline_trainer.py --export
"""

import argparse
import json
import os
import random
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH      = Path("simran_training.db")
DPO_FILE     = Path("training_dataset.jsonl")
# Use llama-3.1-8b-instant for high-volume generation (higher rate limits)
GROQ_MODEL   = "llama-3.1-8b-instant" 
GROQ_CLIENT  = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", ""),
)

def safe_groq_call(messages: List[Dict], temperature: float = 0.7, json_mode: bool = False):
    """Utility to handle Groq API calls with rate-limit retries."""
    max_retries = 5
    delay = 2
    for i in range(max_retries):
        try:
            params = {
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": temperature,
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            
            return GROQ_CLIENT.chat.completions.create(**params)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate_limit" in err_msg:
                if i == max_retries - 1: raise e
                print(f"  ⚠️ Rate limit hit. Retrying in {delay}s... (Attempt {i+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    return None

# ─────────────────────────────────────────────────────────────────────────────
# PERSONAS & SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
PERSONAS = [
    {"name": "Rahul", "style": "excited traveler, speaks Hinglish, wants adventure", "lang": "hinglish"},
    {"name": "Priya", "style": "cautious budget traveler, speaks English, asks many questions", "lang": "english"},
    {"name": "Sharma ji", "style": "elderly man, speaks only Hindi, wants a family pilgrimage", "lang": "hindi"},
    {"name": "Ananya", "style": "busy professional, speaks Hinglish, short and direct", "lang": "hinglish"},
    {"name": "Vikram", "style": "honeymooner, speaks English, wants a luxury resort", "lang": "english"},
    {"name": "Sonia", "style": "suspicious caller, speaks Hindi, thinks it is a scam", "lang": "hindi"},
    {"name": "Amit", "style": "confused student, speaks Hinglish, low budget", "lang": "hinglish"},
    {"name": "Kavita", "style": "solo female traveler, speaks English, concerned about safety", "lang": "english"},
    {"name": "Rajesh", "style": "bargain hunter, speaks Hinglish, always wants more discount", "lang": "hinglish"},
    {"name": "Sneha", "style": "last-minute traveler, speaks Hinglish, needs options TODAY", "lang": "hinglish"},
]

SCENARIOS = [
    "Wants to go to Goa for holiday.",
    "Wants to go to Dubai for luxury shopping.",
    "Wants to go to Thailand for a beach trip.",
    "Wants to go to Bali for a honeymoon.",
    "Customer is busy and wants a call back later.",
    "Customer is not interested and says No.",
    "Customer gives budget but no destination yet.",
    "Customer gives destination but budget is too low.",
    "Customer asks about the website tripinminutes.com.",
    "Customer is confused and doesn't know where to go.",
    "Customer is angry because they got a previous call.",
    "Customer wants a domestic pilgrimage to Kedarnath.",
]

# ─────────────────────────────────────────────────────────────────────────────
# SIMRAN'S SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SIMRAN_SYSTEM = """आप Simran हैं। Trip in Minutes में काम करती हैं।
आपका काम: Customer से 1 Destination और 1 Budget (per person) पता करना।
Style: Warm, respectful, typical Indian woman.
Rules:
- 2 short sentences max. 
- Use "aap" (respect). No "ji" repeatedly.
- matching language (Hindi/English/Hinglish).
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_turn(session_id: str, stage: str, lang: str, cmsg: str, sreply: str, intent: str, score: float, reason: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO conversations 
        (session_id, timestamp, stage, language, customer_msg, simran_reply, intent_detected, score, score_reason, scored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id, datetime.now().isoformat(), stage, lang,
        cmsg, sreply, intent, score, reason, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic(n: int = 50):
    print(f"🚀 Generating {n} synthetic Good/Bad pairs...")
    for i in range(n):
        persona  = random.choice(PERSONAS)
        scenario = random.choice(SCENARIOS)
        
        prompt = f"""Generate a training case for Simran (travel telecaller).
Persona: {persona['style']}
Scenario: {scenario}

Output JSON format ONLY:
{{
  "stage": "greeting/got_interest/got_destination/done",
  "language": "{persona['lang']}",
  "customer_msg": "...",
  "good_reply": "...",
  "bad_reply": "...",
  "intent": "yes/no/busy/soft_no/has_budget",
  "reason": "..."
}}
"""
        try:
            resp = safe_groq_call([{"role": "user", "content": prompt}], temperature=0.8, json_mode=True)
            if not resp: continue
            data = json.loads(resp.choices[0].message.content)
            if isinstance(data, list): data = data[0]
            
            save_turn(f"syn_{i}", data.get('stage', 'unclear'), data.get('language', 'hinglish'), data.get('customer_msg', ''), 
                      data.get('good_reply', ''), data.get('intent', 'unclear'), 9.5, "Synthetic Golden Example")
            save_turn(f"syn_{i}_bad", data.get('stage', 'unclear'), data.get('language', 'hinglish'), data.get('customer_msg', ''), 
                      data.get('bad_reply', ''), data.get('intent', 'unclear'), 3.0, "Synthetic Bad Example: " + data.get('reason', ''))
            
            if (i+1) % 5 == 0: print(f"  Generated {i+1}/{n}...")
            time.sleep(0.05)
        except Exception as e:
            print(f"Error at {i}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: SELF-PLAY SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
def run_selfplay(n: int = 10):
    print(f"🎮 Running {n} self-play conversations...")
    for i in range(n):
        session_id = f"selfplay_{i}_{int(time.time())}"
        persona = random.choice(PERSONAS)
        scenario = random.choice(SCENARIOS)
        
        print(f"  [{i+1}/{n}] Start: {persona['name']} | Status: {scenario}")
        
        history = []
        stage = "greeting"
        
        for turn in range(5):
            # 1. Customer speaks
            customer_prompt = f"""You are {persona['name']}, profile: {persona['style']}.
Scenario: {scenario}
Earlier Conversation: {history}
Goal: Act natural, short message. Focus on {persona['lang']}.
Output only message text."""
            
            cust_call = safe_groq_call([{"role": "user", "content": customer_prompt}], temperature=0.7)
            if not cust_call: break
            cust_resp = cust_call.choices[0].message.content.strip()
            
            # 2. Simran replies
            simran_prompt = f"{SIMRAN_SYSTEM}\nConversation so far: {history}\nCustomer just said: {cust_resp}"
            simran_call = safe_groq_call([{"role": "user", "content": simran_prompt}], temperature=0.4)
            if not simran_call: break
            simran_resp = simran_call.choices[0].message.content.strip()
            
            # 3. Score turn
            score_prompt = f"""Score Simran's reply for quality (0-10).
Customer: {cust_resp}
Simran: {simran_resp}
Format JSON: {{"score": 8.5, "reason": "...", "intent": "..."}}"""
            
            score_call = safe_groq_call([{"role": "user", "content": score_prompt}], json_mode=True)
            if not score_call: break
            score_data = json.loads(score_call.choices[0].message.content)
            
            save_turn(session_id, stage, persona['lang'], cust_resp, simran_resp, 
                      score_data.get('intent', 'unclear'), score_data['score'], score_data['reason'])
            
            history.append(f"Customer: {cust_resp}")
            history.append(f"Simran: {simran_resp}")
            
            if "budget" in simran_resp.lower() or "hazaar" in simran_resp.lower(): stage = "got_destination"
            if "consultant" in simran_resp.lower() or "shukriya" in simran_resp.lower(): break
            
        time.sleep(0.1)

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: CURRICULUM TRAINING
# ─────────────────────────────────────────────────────────────────────────────
LEVELS = {
    1: "Cooperative customer.",
    2: "Normal Hinglish customer.",
    3: "Hesitant customer.",
    4: "Confused customer.",
    5: "Difficult/Busy customer.",
}

def run_curriculum(levels: List[int], n_per_level: int = 10):
    for lv in levels:
        desc = LEVELS.get(lv, "")
        print(f"📚 Level {lv}: {desc}")
        for i in range(n_per_level):
            persona = random.choice(PERSONAS)
            prompt = f"""Generate a conversation turn.
Difficulty: {lv} ({desc})
Persona: {persona['style']}
Output JSON: {{"stage": "...", "customer_msg": "...", "good_reply": "...", "bad_reply": "...", "intent": "..."}}"""
            
            try:
                resp = safe_groq_call([{"role": "user", "content": prompt}], json_mode=True)
                if not resp: continue
                data = json.loads(resp.choices[0].message.content)
                if isinstance(data, list): data = data[0]

                save_turn(f"curr_L{lv}_{i}", data.get('stage', 'unclear'), persona['lang'], data.get('customer_msg', ''), 
                          data.get('good_reply', ''), data.get('intent', 'unclear'), 9.8, f"Curriculum L{lv} Gold")
                save_turn(f"curr_L{lv}_{i}_bad", data.get('stage', 'unclear'), persona['lang'], data.get('customer_msg', ''), 
                          data.get('bad_reply', ''), data.get('intent', 'unclear'), 2.0, f"Curriculum L{lv} Fail")
            except: pass
        print(f"  Level {lv} complete.")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT DPO DATASET
# ─────────────────────────────────────────────────────────────────────────────
def export_dpo():
    print(f"📦 Exporting DPO dataset to {DPO_FILE}...")
    conn = sqlite3.connect(DB_PATH)
    bad_turns = conn.execute("SELECT customer_msg, simran_reply, language, stage FROM conversations WHERE score <= 4").fetchall()
    
    pairs = 0
    with open(DPO_FILE, "w", encoding="utf-8") as f:
        for cmsg, rejected, lang, stage in bad_turns:
            chosen = conn.execute("SELECT simran_reply FROM conversations WHERE customer_msg = ? AND score >= 8 LIMIT 1", (cmsg,)).fetchone()
            if chosen:
                prompt = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                    f"You are Simran, a travel telecaller at Trip in Minutes. "
                    f"Language: {lang}. Stage: {stage}."
                    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                    f"{cmsg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                )
                f.write(json.dumps({
                    "prompt": prompt,
                    "chosen": chosen[0],
                    "rejected": rejected
                }, ensure_ascii=False) + "\n")
                pairs += 1
                
    conn.close()
    print(f"✅ Exported {pairs} pairs. Run 'python trainer.py --train' to start fine-tuning.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simran Offline Training System")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--selfplay", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    
    args = parser.parse_args()
    
    if args.all:
        try:
            generate_synthetic(n=args.n)
            run_selfplay(n=args.n // 5)
            run_curriculum(levels=args.levels, n_per_level=5)
            export_dpo()
        except KeyboardInterrupt:
            print("\n⏹ Stopped by user. Partial data saved to DB.")
    elif args.synthetic: generate_synthetic(n=args.n)
    elif args.selfplay:  run_selfplay(n=args.n)
    elif args.curriculum: run_curriculum(levels=args.levels)
    elif args.export:    export_dpo()
    else: parser.print_help()
