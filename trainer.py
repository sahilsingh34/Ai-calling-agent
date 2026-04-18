"""
trainer.py — Simran Self-Training System
=========================================
Three-stage pipeline that makes Simran automatically improve:

Stage 1: LOG    — Every call is logged with full context to SQLite
Stage 2: SCORE  — Groq auto-scores each response (RLAIF — no humans needed)
Stage 3: TRAIN  — DPO fine-tuning on good/bad pairs using Hugging Face TRL
Stage 4: DEPLOY — LoRA adapter hot-swapped into production

How to run:
  python trainer.py --score          # Score all unscored conversations
  python trainer.py --export         # Export training dataset
  python trainer.py --train          # Fine-tune the model
  python trainer.py --dynamic-test   # Test dynamic few-shot injection
  python trainer.py --full-cycle     # Run all stages end to end

Requirements:
  pip install transformers trl peft bitsandbytes datasets torch openai
"""

import argparse
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH          = Path("simran_training.db")
DATASET_PATH     = Path("training_dataset.jsonl")
ADAPTER_PATH     = Path("simran_adapter")
MIN_SCORE_GOOD   = 7      # Score >= 7 = "chosen" (good) example
MAX_SCORE_BAD    = 4      # Score <= 4 = "rejected" (bad) example
MIN_PAIRS_TRAIN  = 50     # Minimum pairs needed before fine-tuning
BASE_MODEL       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GROQ_MODEL       = "llama-3.3-70b-versatile"
MAX_EXAMPLES_FEW_SHOT = 5  # How many best examples to inject into prompt

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", ""),
)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────
def init_training_db():
    """Create all training tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Every conversation turn gets logged here
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            stage       TEXT,              -- greeting/got_interest/got_destination/done
            language    TEXT,              -- hindi/hinglish/english
            customer_msg TEXT NOT NULL,
            simran_reply TEXT NOT NULL,
            intent_detected TEXT,          -- no/soft_no/yes/busy/greeting_only/unclear
            dest_collected TEXT,           -- destination if found
            budget_collected TEXT,         -- budget if found
            call_outcome TEXT,             -- lead/no/busy/soft_no
            score       REAL DEFAULT NULL, -- AI score 0-10 (set after scoring)
            score_reason TEXT DEFAULT NULL,-- why this score was given
            scored_at   TEXT DEFAULT NULL
        )
    """)

    # Best examples for dynamic few-shot injection
    c.execute("""
        CREATE TABLE IF NOT EXISTS few_shot_examples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            language    TEXT NOT NULL,     -- which language context this example is for
            stage       TEXT NOT NULL,     -- which stage this example is for
            customer_msg TEXT NOT NULL,
            simran_reply TEXT NOT NULL,
            score       REAL NOT NULL,
            used_count  INTEGER DEFAULT 0, -- how many times injected
            created_at  TEXT NOT NULL
        )
    """)

    # Training run history
    c.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at      TEXT NOT NULL,
            completed_at    TEXT,
            pairs_used      INTEGER,
            base_model      TEXT,
            adapter_path    TEXT,
            avg_score_before REAL,
            avg_score_after  REAL,
            status          TEXT DEFAULT 'running'
        )
    """)

    conn.commit()
    conn.close()
    print("Training database ready.")


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — called from main.py after every reply
# ─────────────────────────────────────────────────────────────────────────────
def log_conversation(
    session_id: str,
    stage: str,
    language: str,
    customer_msg: str,
    simran_reply: str,
    intent_detected: str,
    dest_collected: str = "",
    budget_collected: str = "",
    call_outcome: str = "",
):
    """
    Log one conversation turn to the training database.
    Call this from process_speech() after every reply is generated.

    Usage in main.py (add after history.append):
        from trainer import log_conversation
        log_conversation(
            session_id=self.stream_id or "demo",
            stage=self.stage,
            language=self.customer_language,
            customer_msg=text,
            simran_reply=full,
            intent_detected=intent,
            dest_collected=self.lead_destination,
            budget_collected=self.lead_budget,
        )
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO conversations
            (session_id, timestamp, stage, language, customer_msg, simran_reply,
             intent_detected, dest_collected, budget_collected, call_outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now().isoformat(),
            stage, language,
            customer_msg[:1000],
            simran_reply[:1000],
            intent_detected,
            dest_collected,
            budget_collected,
            call_outcome,
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[trainer] Log failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: AUTO-SCORING (RLAIF)
# Uses Groq to judge each response on 10 criteria specific to Simran's job
# ─────────────────────────────────────────────────────────────────────────────
SCORING_PROMPT = """You are an expert quality evaluator for Simran, an Indian travel telecaller at Trip in Minutes.

Simran's ONLY job: collect destination + budget from the customer, then close warmly.
Simran speaks simple Hinglish/Hindi/English matching the customer.
Simran uses "aap" (never "tum"), never says "ji" repeatedly, sounds like a real Indian woman.

Rate this response from 0-10:

Customer said: "{customer_msg}"
Simran replied: "{simran_reply}"
Stage: {stage}
Language: {language}
Destination already collected: "{dest_collected}"
Budget already collected: "{budget_collected}"

Scoring criteria (each worth up to 1 point):
1. Matches customer's language (Hindi→Hindi, English→English, Hinglish→Hinglish)
2. Uses "aap" correctly, respectful tone
3. Acknowledges what customer said before responding
4. Asks exactly ONE question, not more
5. Does NOT repeat info customer already gave
6. Moves conversation forward toward destination or budget
7. Sounds natural, not scripted or robotic
8. Appropriate length (max 2 short sentences for phone call)
9. Warm, friendly tone without being pushy
10. Correct stage logic (if dest given, asks budget; if both given, closes warmly)

Respond ONLY with this JSON (no other text):
{{"score": <number 0-10>, "reason": "<one sentence why>"}}"""


def score_unscored_conversations(batch_size: int = 50) -> int:
    """
    Score all conversations that haven't been scored yet.
    Returns number of conversations scored.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, customer_msg, simran_reply, stage, language,
               dest_collected, budget_collected
        FROM conversations
        WHERE score IS NULL
        ORDER BY id ASC
        LIMIT ?
    """, (batch_size,)).fetchall()

    if not rows:
        print("No unscored conversations found.")
        conn.close()
        return 0

    scored = 0
    for row in rows:
        conv_id, customer_msg, simran_reply, stage, language, dest, budget = row

        prompt = SCORING_PROMPT.format(
            customer_msg=customer_msg,
            simran_reply=simran_reply,
            stage=stage or "greeting",
            language=language or "hinglish",
            dest_collected=dest or "none",
            budget_collected=budget or "none",
        )

        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            score  = float(data.get("score", 5))
            reason = str(data.get("reason", ""))

            conn.execute("""
                UPDATE conversations
                SET score=?, score_reason=?, scored_at=?
                WHERE id=?
            """, (score, reason, datetime.now().isoformat(), conv_id))
            scored += 1

            if scored % 10 == 0:
                print(f"  Scored {scored}/{len(rows)}...")

            time.sleep(0.2)  # Rate limit respect

        except Exception as e:
            print(f"  Scoring failed for id={conv_id}: {e}")
            continue

    conn.commit()

    # Promote high-scoring examples to few_shot_examples table
    good = conn.execute("""
        SELECT stage, language, customer_msg, simran_reply, score
        FROM conversations
        WHERE score >= ? AND scored_at IS NOT NULL
        ORDER BY score DESC
        LIMIT 100
    """, (MIN_SCORE_GOOD,)).fetchall()

    few_shot_added = 0
    for stage, language, cmsg, sreply, score in good:
        existing = conn.execute("""
            SELECT id FROM few_shot_examples
            WHERE customer_msg = ? AND simran_reply = ?
        """, (cmsg, sreply)).fetchone()
        if not existing:
            conn.execute("""
                INSERT INTO few_shot_examples
                (language, stage, customer_msg, simran_reply, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (language or "hinglish", stage or "greeting",
                  cmsg, sreply, score, datetime.now().isoformat()))
            few_shot_added += 1

    conn.commit()
    conn.close()

    print(f"Scored {scored} conversations. Added {few_shot_added} new few-shot examples.")
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC FEW-SHOT — inject best examples into every LLM prompt
# This is the FASTEST way to improve responses with zero training cost
# ─────────────────────────────────────────────────────────────────────────────
def get_few_shot_examples(
    language: str = "hinglish",
    stage: str = "greeting",
    n: int = MAX_EXAMPLES_FEW_SHOT,
) -> list[dict]:
    """
    Fetch the best N examples for this language + stage combination.
    These get injected into the LLM prompt as demonstrations.

    Usage in main.py — add to _answer_question and _llm_nudge:
        from trainer import get_few_shot_examples
        examples = get_few_shot_examples(language=self.customer_language, stage=self.stage)
        # Inject into messages as few-shot examples before the user message
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT customer_msg, simran_reply, score
            FROM few_shot_examples
            WHERE language = ? AND stage = ?
            ORDER BY score DESC, used_count ASC
            LIMIT ?
        """, (language, stage, n)).fetchall()

        if len(rows) < 2:
            # Fall back to any language for this stage
            rows = conn.execute("""
                SELECT customer_msg, simran_reply, score
                FROM few_shot_examples
                WHERE stage = ?
                ORDER BY score DESC
                LIMIT ?
            """, (stage, n)).fetchall()

        # Increment usage count
        if rows:
            for row in rows:
                conn.execute("""
                    UPDATE few_shot_examples
                    SET used_count = used_count + 1
                    WHERE customer_msg = ? AND simran_reply = ?
                """, (row[0], row[1]))
            conn.commit()

        conn.close()

        examples = []
        for cmsg, sreply, score in rows:
            examples.append({"role": "user", "content": cmsg})
            examples.append({"role": "assistant", "content": sreply})
        return examples

    except Exception as e:
        print(f"[trainer] Few-shot fetch failed: {e}")
        return []


def build_few_shot_header(examples: list[dict]) -> str:
    """Convert few-shot examples to a text block for the system prompt."""
    if not examples:
        return ""
    lines = ["\n\nBEST EXAMPLES FROM REAL CALLS (follow this style exactly):"]
    for i in range(0, len(examples), 2):
        if i + 1 < len(examples):
            lines.append(f"\nCustomer: {examples[i]['content']}")
            lines.append(f"Simran: {examples[i+1]['content']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: EXPORT TRAINING DATASET
# Creates good/bad pairs in DPO format for fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
def export_training_dataset() -> int:
    """
    Export scored conversations as DPO preference pairs.
    Format: {prompt, chosen, rejected} — standard for TRL DPO training.
    Returns number of pairs exported.
    """
    conn = sqlite3.connect(DB_PATH)

    good = conn.execute("""
        SELECT stage, language, customer_msg, simran_reply, score, score_reason
        FROM conversations
        WHERE score >= ?
        ORDER BY score DESC
    """, (MIN_SCORE_GOOD,)).fetchall()

    bad = conn.execute("""
        SELECT stage, language, customer_msg, simran_reply, score, score_reason
        FROM conversations
        WHERE score <= ?
        ORDER BY score ASC
    """, (MAX_SCORE_BAD,)).fetchall()

    conn.close()

    if len(good) < 10 or len(bad) < 10:
        print(f"Not enough data yet. Have {len(good)} good, {len(bad)} bad examples.")
        print(f"Need at least 10 of each. Keep logging more calls!")
        return 0

    # Build pairs — each bad example gets matched with a good example from same stage
    pairs = []
    bad_by_stage = {}
    for stage, lang, cmsg, sreply, score, reason in bad:
        key = (stage or "greeting", lang or "hinglish")
        if key not in bad_by_stage:
            bad_by_stage[key] = []
        bad_by_stage[key].append((cmsg, sreply, score))

    for stage, lang, cmsg, chosen_reply, score, reason in good:
        key = (stage or "greeting", lang or "hinglish")
        # Find a bad example for the same customer message context
        bad_pool = bad_by_stage.get(key, []) or bad_by_stage.get(
            (stage or "greeting", "hinglish"), []
        )
        if not bad_pool:
            continue

        # Match by finding a bad reply to a similar-stage customer message
        rejected_reply = bad_pool[len(pairs) % len(bad_pool)][1]

        # DPO format — Llama 3.1 chat template
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are Simran, a travel telecaller at Trip in Minutes. "
            f"Language: {lang}. Stage: {stage}."
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{cmsg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        pairs.append({
            "prompt":   prompt,
            "chosen":   chosen_reply,
            "rejected": rejected_reply,
            "stage":    stage,
            "language": lang,
            "score_chosen":   score,
        })

    if len(pairs) < MIN_PAIRS_TRAIN:
        print(f"Only {len(pairs)} pairs available. Need {MIN_PAIRS_TRAIN} to train.")
        print("Keep running calls and scoring — come back when you have more data.")
        return len(pairs)

    # Write JSONL
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Exported {len(pairs)} training pairs to {DATASET_PATH}")
    return len(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: DPO FINE-TUNING
# Uses Hugging Face TRL + LoRA — trains on your preference pairs
# This makes Simran permanently better — no prompt changes needed
# ─────────────────────────────────────────────────────────────────────────────
def run_fine_tuning():
    """
    Fine-tune Llama 3.1 8B with DPO using LoRA.
    Requires: pip install transformers trl peft bitsandbytes datasets torch
    GPU recommended (works on single 16GB GPU with 4-bit quantization).

    After training, the LoRA adapter is saved to ADAPTER_PATH.
    Deploy by uploading to Groq or serving locally with vLLM.
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        print("""
Fine-tuning libraries not installed. Run:
  pip install transformers trl peft bitsandbytes datasets torch

Note: GPU with 16GB+ VRAM recommended.
For CPU-only testing, use the dynamic few-shot method instead.
        """)
        return

    if not DATASET_PATH.exists():
        print("No training dataset found. Run --export first.")
        return

    # Count pairs
    with open(DATASET_PATH) as f:
        n_pairs = sum(1 for _ in f)

    if n_pairs < MIN_PAIRS_TRAIN:
        print(f"Only {n_pairs} pairs. Need {MIN_PAIRS_TRAIN}. Keep logging calls.")
        return

    print(f"Starting DPO fine-tuning on {n_pairs} pairs...")
    print(f"Base model: {BASE_MODEL}")

    # Log training run
    conn = sqlite3.connect(DB_PATH)
    run_id = conn.execute("""
        INSERT INTO training_runs (started_at, pairs_used, base_model, status)
        VALUES (?, ?, ?, 'running')
    """, (datetime.now().isoformat(), n_pairs, BASE_MODEL)).lastrowid
    conn.commit()

    avg_before = conn.execute(
        "SELECT AVG(score) FROM conversations WHERE score IS NOT NULL"
    ).fetchone()[0] or 0
    conn.close()

    try:
        # 4-bit quantization — only if CUDA available (bitsandbytes req)
        bnb_config = None
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            print("🚀 CUDA not found. Loading model in full/half precision (CPU/MPS friendly).")

        print("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # LoRA config — small adapters, fast training
        lora_config = LoraConfig(
            r=16,                          # rank — higher = more capacity
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load dataset
        dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
        # 90% train, 10% eval
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

        # DPO training config
        training_args = DPOConfig(
            output_dir=str(ADAPTER_PATH),
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            beta=0.1,                      # DPO temperature — controls how strongly
                                           # the model prefers good over bad examples
            max_length=512,
            max_prompt_length=256,
            report_to="none",
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,                # DPO with implicit reference
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
        )

        print("Training started...")
        trainer.train()

        # Save the LoRA adapter
        ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(ADAPTER_PATH))
        tokenizer.save_pretrained(str(ADAPTER_PATH))

        print(f"\nTraining complete! Adapter saved to: {ADAPTER_PATH}")
        print(f"Upload this adapter to Groq or serve with vLLM.")

        # Update training log
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            UPDATE training_runs
            SET completed_at=?, adapter_path=?, avg_score_before=?, status='completed'
            WHERE id=?
        """, (datetime.now().isoformat(), str(ADAPTER_PATH), avg_before, run_id))
        conn.commit()
        conn.close()

    except Exception as e:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE training_runs SET status='failed' WHERE id=?", (run_id,)
        )
        conn.commit()
        conn.close()
        print(f"Training failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STATS — see how Simran is improving over time
# ─────────────────────────────────────────────────────────────────────────────
def print_stats():
    """Print a dashboard of Simran's performance over time."""
    conn = sqlite3.connect(DB_PATH)

    total   = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    scored  = conn.execute("SELECT COUNT(*) FROM conversations WHERE score IS NOT NULL").fetchone()[0]
    avg     = conn.execute("SELECT AVG(score) FROM conversations WHERE score IS NOT NULL").fetchone()[0]
    good    = conn.execute("SELECT COUNT(*) FROM conversations WHERE score >= ?", (MIN_SCORE_GOOD,)).fetchone()[0]
    bad     = conn.execute("SELECT COUNT(*) FROM conversations WHERE score <= ?", (MAX_SCORE_BAD,)).fetchone()[0]
    fewshot = conn.execute("SELECT COUNT(*) FROM few_shot_examples").fetchone()[0]
    runs    = conn.execute("SELECT COUNT(*) FROM training_runs WHERE status='completed'").fetchone()[0]

    # Score trend (last 3 weeks)
    trend = conn.execute("""
        SELECT DATE(timestamp) as day, AVG(score) as avg_score, COUNT(*) as calls
        FROM conversations
        WHERE score IS NOT NULL AND timestamp >= DATE('now', '-21 days')
        GROUP BY day
        ORDER BY day ASC
    """).fetchall()

    # Best performing stages
    by_stage = conn.execute("""
        SELECT stage, AVG(score) as avg, COUNT(*) as n
        FROM conversations
        WHERE score IS NOT NULL
        GROUP BY stage
        ORDER BY avg DESC
    """).fetchall()

    # Language breakdown
    by_lang = conn.execute("""
        SELECT language, AVG(score) as avg, COUNT(*) as n
        FROM conversations
        WHERE score IS NOT NULL
        GROUP BY language
        ORDER BY avg DESC
    """).fetchall()

    conn.close()

    print("\n" + "="*55)
    print("  SIMRAN PERFORMANCE DASHBOARD")
    print("="*55)
    print(f"  Total conversations logged : {total}")
    print(f"  Scored conversations       : {scored}")
    print(f"  Average quality score      : {avg:.1f}/10" if avg else "  Average score: N/A")
    print(f"  Good examples (>={MIN_SCORE_GOOD})      : {good}")
    print(f"  Bad examples (<={MAX_SCORE_BAD})       : {bad}")
    print(f"  Few-shot bank size         : {fewshot} examples")
    print(f"  Completed training runs    : {runs}")

    if trend:
        print(f"\n  Score trend (last 21 days):")
        for day, avg_s, calls in trend[-7:]:
            bar = "█" * int(avg_s)
            print(f"    {day}  {avg_s:.1f}/10  {bar}  ({calls} calls)")

    if by_stage:
        print(f"\n  Performance by stage:")
        for stage, avg_s, n in by_stage:
            print(f"    {(stage or 'unknown'):<20} {avg_s:.1f}/10  ({n} samples)")

    if by_lang:
        print(f"\n  Performance by language:")
        for lang, avg_s, n in by_lang:
            print(f"    {(lang or 'unknown'):<12} {avg_s:.1f}/10  ({n} samples)")

    print("="*55)

    # Training recommendation
    if good >= MIN_PAIRS_TRAIN and bad >= MIN_PAIRS_TRAIN:
        print(f"\n  READY TO TRAIN: You have enough data!")
        print(f"  Run: python trainer.py --train")
    elif scored < 20:
        print(f"\n  Keep logging calls. Score more with: python trainer.py --score")
    else:
        need_more = max(0, MIN_PAIRS_TRAIN - min(good, bad))
        print(f"\n  Need {need_more} more scored conversations before training.")


# ─────────────────────────────────────────────────────────────────────────────
# SEED EXAMPLES — bootstrap the few-shot bank with hand-crafted gold examples
# Run this once when you first set up the system
# ─────────────────────────────────────────────────────────────────────────────
GOLD_EXAMPLES = [
    # Hinglish examples
    {
        "language": "hinglish", "stage": "greeting",
        "customer_msg": "haan boliye",
        "simran_reply": "Hello! Kaise hain aap? Koi trip plan chal rahi hai kya?",
        "score": 9.5,
    },
    {
        "language": "hinglish", "stage": "got_interest",
        "customer_msg": "thailand jaana hai",
        "simran_reply": "Thailand bahut sundar jagah hai! Per person kitna budget soch rahe hain roughly?",
        "score": 9.5,
    },
    {
        "language": "hinglish", "stage": "got_interest",
        "customer_msg": "goa jaana chahte hain",
        "simran_reply": "Goa! Wahan bahut maza aata hai, beaches aur khana dono amazing. Budget roughly kitna soch rahe hain aap?",
        "score": 9.8,
    },
    {
        "language": "hinglish", "stage": "got_destination",
        "customer_msg": "50 hazaar ke aaspaas",
        "simran_reply": "Bahut accha! Hamare consultant aapko Goa ke kuch acche options ke saath jald call karenge. Aapka time dene ka shukriya!",
        "score": 9.5,
    },
    {
        "language": "hinglish", "stage": "greeting",
        "customer_msg": "abhi nahi karna par baad mein jaaunga",
        "simran_reply": "Bilkul, sochiye aap! Main bas itna batati hoon — popular jagahon ke liye 30-50 hazaar mein bahut acche options hain. Koi sawal ho toh tripinminutes.com pe dekh sakte hain.",
        "score": 9.0,
    },
    # Hindi (Devanagari) examples
    {
        "language": "hindi", "stage": "got_interest",
        "customer_msg": "थाईलैंड जाने का सोच रहा हूं",
        "simran_reply": "Thailand bahut sundar jagah hai! Per person roughly kitna budget soch rahe hain aap?",
        "score": 9.2,
    },
    {
        "language": "hindi", "stage": "greeting",
        "customer_msg": "हेलो",
        "simran_reply": "Hello! Kaise hain aap? Koi trip plan chal rahi hai kya?",
        "score": 9.0,
    },
    # English examples
    {
        "language": "english", "stage": "got_interest",
        "customer_msg": "I want to go to Bali",
        "simran_reply": "Bali is such a beautiful place! What's your rough budget per person?",
        "score": 9.5,
    },
    {
        "language": "english", "stage": "greeting",
        "customer_msg": "hello",
        "simran_reply": "Hello! How are you? Are you planning any trips these days?",
        "score": 9.0,
    },
    {
        "language": "english", "stage": "got_destination",
        "customer_msg": "maybe around 60000",
        "simran_reply": "Perfect! Our travel consultant will call you shortly with the best Thailand options. Thank you for your time, have a great day!",
        "score": 9.5,
    },
]


def seed_gold_examples():
    """
    Seed the few-shot bank with hand-crafted perfect examples.
    Run once at startup: python trainer.py --seed
    """
    conn = sqlite3.connect(DB_PATH)
    added = 0
    for ex in GOLD_EXAMPLES:
        existing = conn.execute(
            "SELECT id FROM few_shot_examples WHERE customer_msg=? AND simran_reply=?",
            (ex["customer_msg"], ex["simran_reply"])
        ).fetchone()
        if not existing:
            conn.execute("""
                INSERT INTO few_shot_examples
                (language, stage, customer_msg, simran_reply, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ex["language"], ex["stage"],
                ex["customer_msg"], ex["simran_reply"],
                ex["score"], datetime.now().isoformat()
            ))
            added += 1
    conn.commit()
    conn.close()
    print(f"Seeded {added} gold examples into few-shot bank.")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION HELPERS — paste these into main.py
# ─────────────────────────────────────────────────────────────────────────────
MAIN_PY_INTEGRATION_SNIPPET = '''
# ══════════════════════════════════════════════════════════
# ADD TO main.py — Self-Training Integration
# ══════════════════════════════════════════════════════════

# 1. At the top of main.py, add:
from trainer import log_conversation, get_few_shot_examples, build_few_shot_header

# 2. In CallSession.process_speech(), after building the full reply:
#    (add this after: self.history.append({"role": "assistant", "content": full}))

    log_conversation(
        session_id  = self.stream_id or "live",
        stage       = self.stage,
        language    = self.customer_language,
        customer_msg= text,
        simran_reply= full,
        intent_detected = classify_intent(text),
        dest_collected  = self.lead_destination,
        budget_collected= self.lead_budget,
        call_outcome    = "lead" if self.stage == "done" and self.lead_budget else self.stage,
    )

# 3. In _answer_question() and _llm_nudge(), inject few-shot examples:
#    (add this before building the messages list)

    examples = get_few_shot_examples(
        language=self.customer_language,
        stage=self.stage,
        n=3,
    )
    few_shot_block = build_few_shot_header(examples)
    
    # Then append few_shot_block to the system prompt:
    system_with_examples = SIMRAN_SYSTEM + few_shot_block
    
    messages = [
        {"role": "system", "content": system_with_examples},
        *self.history[-HISTORY_WINDOW:],
        {"role": "user", "content": ...}
    ]
'''


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simran Self-Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  --init          Create database and seed gold examples (run once)
  --score         Score all unscored conversations with AI
  --export        Export training dataset (good vs bad pairs)
  --train         Run DPO fine-tuning (needs GPU + libraries)
  --stats         Show performance dashboard
  --full-cycle    Score + export + train in one go
  --integration   Print the code to paste into main.py
        """
    )
    parser.add_argument("--init",        action="store_true", help="Initialize DB and seed examples")
    parser.add_argument("--score",       action="store_true", help="Score unscored conversations")
    parser.add_argument("--export",      action="store_true", help="Export training dataset")
    parser.add_argument("--train",       action="store_true", help="Run DPO fine-tuning")
    parser.add_argument("--stats",       action="store_true", help="Show stats dashboard")
    parser.add_argument("--full-cycle",  action="store_true", help="Score + export + train")
    parser.add_argument("--integration", action="store_true", help="Print main.py integration code")
    parser.add_argument("--batch",       type=int, default=50, help="Batch size for scoring")
    args = parser.parse_args()

    # Always ensure DB exists
    init_training_db()

    if args.init:
        seed_gold_examples()
        print_stats()

    elif args.score:
        n = score_unscored_conversations(batch_size=args.batch)
        print_stats()

    elif args.export:
        export_training_dataset()

    elif args.train:
        run_fine_tuning()

    elif args.stats:
        print_stats()

    elif args.full_cycle:
        print("=== FULL TRAINING CYCLE ===")
        print("\n[1/3] Scoring conversations...")
        score_unscored_conversations(batch_size=200)
        print("\n[2/3] Exporting dataset...")
        n_pairs = export_training_dataset()
        if n_pairs >= MIN_PAIRS_TRAIN:
            print("\n[3/3] Running fine-tuning...")
            run_fine_tuning()
        else:
            print(f"\n[3/3] Skipping training — only {n_pairs} pairs (need {MIN_PAIRS_TRAIN})")
        print_stats()

    elif args.integration:
        print(MAIN_PY_INTEGRATION_SNIPPET)

    else:
        parser.print_help()
        print("\nRun --stats to see current performance.")
        print_stats()