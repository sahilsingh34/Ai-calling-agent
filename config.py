import os
from dotenv import load_dotenv
load_dotenv()

# =============================================================
#  TRIP IN MINUTES — SIMRAN TELECALLER CONFIG v7.0 PRODUCTION
# =============================================================

AGENT_NAME   = "Simran"
COMPANY_NAME = "Trip in Minutes"
COMPANY_URL  = "tripinminutes.com"

fallback_greeting = (
    "Namaste! Main Simran bol rahi hoon, Trip in Minutes se. "
    "Kya aap koi trip ya vacation plan kar rahe hain?"
)

# ── STT — Deepgram ─────────────────────────────────────────
# language=multi: best for Hindi-English code-switching
# endpointing=1000: natural pause detection (300 was too aggressive)
# utterance_end_ms=1500: buffer for long pauses mid-sentence
DEEPGRAM_API_KEY     = os.getenv("DEEPGRAM_API_KEY", "")
STT_MODEL            = "nova-2"
STT_LANGUAGE         = "multi"
STT_ENDPOINTING      = 1000
STT_UTTERANCE_END_MS = 1500

# ── TTS — Cartesia ──────────────────────────────────────────
CARTESIA_API_KEY  = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VERSION  = "2025-04-16"
CARTESIA_MODEL    = "sonic-3"
CARTESIA_VOICE_ID = "47f3bbb1-e98f-4e0c-92c5-5f0325e1e206"
CARTESIA_LANGUAGE = "hi"

CARTESIA_OUTPUT_FORMAT = {
    "container":   "raw",
    "encoding":    "pcm_mulaw",
    "sample_rate": 8000,
}
CARTESIA_TEST_FORMAT = {
    "container":   "wav",
    "encoding":    "pcm_f32le",
    "sample_rate": 44100,
}

# ── LLM — Groq ──────────────────────────────────────────────
# max_tokens=50: faster first word, forces short replies
# temperature=0.35: consistent but not robotic
# LLM_CALL_TIMEOUT: hard limit — fallback response after this
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL        = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE  = 0.35
GROQ_MAX_TOKENS   = 50
LLM_CALL_TIMEOUT  = 5.0   # seconds — fallback if Groq is slow

DEFAULT_LLM_MODEL = GROQ_MODEL

# ── Telephony — Vobiz ───────────────────────────────────────
VOBIZ_AUTH_ID     = os.getenv("VOBIZ_AUTH_ID", "")
VOBIZ_AUTH_TOKEN  = os.getenv("VOBIZ_AUTH_TOKEN", "")
VOBIZ_FROM_NUMBER = os.getenv("VOBIZ_FROM_NUMBER", "")

# ── Timing ──────────────────────────────────────────────────
POST_SPEECH_DELAY_MS = 200

# ── Required .env keys ──────────────────────────────────────
# DEEPGRAM_API_KEY=
# CARTESIA_API_KEY=
# GROQ_API_KEY=
# VOBIZ_AUTH_ID=
# VOBIZ_AUTH_TOKEN=
# VOBIZ_FROM_NUMBER=+91xxxxxxxxxx
# PUBLIC_URL=https://your-tunnel.trycloudflare.com
# PORT=5050