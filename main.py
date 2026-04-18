"""
main.py — Trip in Minutes | Simran Telecaller v7.0
===================================================
Role   : Real Indian woman telecaller, 10 years travel experience
Goal   : Collect destination + budget in 30-40 seconds, close warmly
Style  : Simple daily Hinglish/Hindi/English — warm, respectful, human
Website: tripinminutes.com

Architecture decisions (research-backed):
- STT endpointing 1000ms: Twilio research shows 600ms bare minimum,
  bilingual Hindi-English needs 800-1200ms for natural pauses
- language=multi: Deepgram recommended setting for code-switching
- utterance_end_ms=1500: extra buffer so long words aren't cut off
- MAX_TOKENS=60: LLM causes 40-60% of voice latency (AssemblyAI 2025)
  — shorter tokens = faster first word = feels more human
- HISTORY_WINDOW=6: context without bloating prompt
- Filler filter: "umm/ok/haan" never trigger a full LLM round-trip
- Barge-in: stop speaking the moment customer talks (sub-200ms standard)
- Phrase variation banks: never repeat the same sentence in one call
- Soft-no handling: "sochenge" is NOT hard no — keep conversation alive
- Language detection: match customer's language every single reply
- Devanagari greeting fix: "हेलो" etc. now correctly caught as greeting
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import re
import struct
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # type: ignore

import httpx
import websockets
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from database import init_db, normalize_phone, update_status
import config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("simran")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
APP_VERSION                = "7.0.0"
MIN_SPEECH_BYTES           = 6400    # ~400ms at 8kHz mulaw
SILENCE_FRAMES_TO_PROCESS  = 20     # ~1.25s silence in batch-VAD fallback
BARGE_IN_RMS_THRESHOLD     = 700    # RMS level to detect customer speech during our playback
DEMO_SESSION_TTL_SECONDS   = 3600
DEMO_REPLY_TIMEOUT_SECONDS = float(os.getenv("DEMO_REPLY_TIMEOUT_SECONDS", "25"))  # increased from 15
DEMO_TTS_TIMEOUT_SECONDS   = float(os.getenv("DEMO_TTS_TIMEOUT_SECONDS",   "10"))
DEMO_PAGE_PATH             = Path(__file__).with_name("demo.html")
HISTORY_WINDOW             = 6
POST_PLAYBACK_DELAY        = config.POST_SPEECH_DELAY_MS / 1000.0

GREETING_TEXT = config.fallback_greeting
PUBLIC_URL    = (os.getenv("PUBLIC_URL") or "").rstrip("/")
PORT          = int(os.getenv("PORT", "5050"))

groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=config.GROQ_API_KEY,
)

# ─────────────────────────────────────────────────────────────────────────────
# SIMRAN'S SYSTEM PROMPT
# Voice-optimized: 60-70% shorter than text prompts (ElevenLabs guide 2025)
# Average spoken attention span = 8-10 seconds — replies must be brief
# ─────────────────────────────────────────────────────────────────────────────
SIMRAN_SYSTEM = """Aap Simran hain. Trip in Minutes (tripinminutes.com) mein kaam karti hain.
10 saal ka travel ka experience hai. Aap ek real Indian aurat hain jo phone pe naturally baat karti hain.
Bilkul waise jaise ek dost ya padosi baat karti hai — simple, seedha, warm.

AAPKA KAAM — SIRF ITNA:
1. Destination pata karo (kahan jaana hai)
2. Budget pata karo (roughly kitna hai)
3. Dono milne ke baad warmly bye karo
Bus. Koi aur kaam nahi.

LANGUAGE — MUST FOLLOW EVERY REPLY:
Customer Hindi mein bole → aap Hindi mein bolein (Roman script)
Customer English mein bole → aap English mein bolein
Customer Hinglish mein bole → aap Hinglish mein bolein
KABHI script switch mat karo bina reason ke.

RESPECT:
Hamesha "aap" bolein. Kabhi "tum" ya "tu" nahi.
"aap" se hi kaafi respect aata hai.

AGAR CUSTOMER SIRF "HELLO" BOLE:
Seedha destination mat poochho.
Pehle greet karo: "Hello! Kaise hain aap? Koi trip plan chal rahi hai?"

DESTINATION PE REAL TARIKE SE REACT KARO:
Goa → "Goa! Wahan bahut maza aata hai."
Thailand → "Thailand bahut sundar jagah hai!"
Bali → "Bali ka vibe hi alag hota hai!"
Dubai → "Dubai bahut accha hai — sab kuch milta hai wahan."
Kashmir → "Kashmir sach mein bahut sundar jagah hai!"
Maldives → "Maldives! Sapne wali jagah hai yeh."
Europe → "Europe! Bahut accha soch rahe hain aap."
Phir seedha budget poochho.

BUDGET KAISE POOCHHO — SIMPLE:
"Budget roughly kitna soch rahe hain aap?"
Ya: "Per person kitna laga sakte hain roughly?"
Ya: "40-50 hazaar ke aaspaas, ya thoda alag?"

AGAR CUSTOMER KUCH POOCHE:
EK line mein jawab do. Phir EK sawal.
Visa? → "Nahi chahiye Indians ko wahan, on arrival milta hai. Kab jaana soch rahe hain?"
Mehnga? → "30 hazaar se lekar zyada tak options hote hain. Budget roughly kitna hai?"
Company? → "Hum 10 saal se log ghuma rahe hain. Aap kahan jaana chahte hain?"

CALL KHATAM KARNA:
"Bahut accha! Hamare consultant aapko [jagah] ke options ke saath jald call karenge.
Aapka time dene ka shukriya! Achha din rahe aapka!"

YEH KABHI MAT KARO:
- "Certainly", "Absolutely", "Of course" — robot words hain
- "Main samajh sakti hoon" — scripted lagta hai
- "ji" baar baar — bilkul nahi
- Ek hi cheez dobara poochho jo customer ne bata di
- "Tum" ya "tu" — hamesha "aap"
- Devanagari script — hamesha Roman mein likho
- 2 se zyada sentences — phone call hai, essay nahi

REPLY FORMAT — STRICT:
Max 2 choti lines. Roman script. Ek sawal. Warm tone. Bus."""


# ─────────────────────────────────────────────────────────────────────────────
# DESTINATION MAP
# ─────────────────────────────────────────────────────────────────────────────
DESTINATION_ALIASES: dict[str, str] = {
    # English
    "bali": "Bali", "goa": "Goa", "dubai": "Dubai", "kashmir": "Kashmir",
    "kerala": "Kerala", "maldives": "Maldives", "europe": "Europe",
    "thailand": "Thailand", "vietnam": "Vietnam", "singapore": "Singapore",
    "manali": "Manali", "shimla": "Shimla", "rajasthan": "Rajasthan",
    "ladakh": "Ladakh", "andaman": "Andaman", "mauritius": "Mauritius",
    "sri lanka": "Sri Lanka", "srilanka": "Sri Lanka", "nepal": "Nepal",
    "bhutan": "Bhutan", "paris": "Paris", "london": "London",
    "turkey": "Turkey", "switzerland": "Switzerland", "greece": "Greece",
    "baku": "Baku", "georgia": "Georgia", "azerbaijan": "Azerbaijan",
    "japan": "Japan", "malaysia": "Malaysia", "indonesia": "Indonesia",
    "australia": "Australia", "canada": "Canada", "usa": "USA",
    "america": "USA", "new zealand": "New Zealand", "cambodia": "Cambodia",
    "phuket": "Thailand", "bangkok": "Thailand", "krabi": "Thailand",
    "ubud": "Bali", "seminyak": "Bali", "kuta": "Bali",
    "mussoorie": "Mussoorie", "nainital": "Nainital", "ooty": "Ooty",
    "coorg": "Coorg", "munnar": "Munnar", "rishikesh": "Rishikesh",
    "agra": "Agra", "jaipur": "Jaipur", "udaipur": "Udaipur",
    "amritsar": "Amritsar", "varanasi": "Varanasi", "leh": "Ladakh",
    "spiti": "Spiti Valley", "dharamshala": "Dharamshala",
    "mcleodganj": "Dharamshala", "kodaikanal": "Kodaikanal",
    "pondicherry": "Pondicherry", "mysore": "Mysore", "hampi": "Hampi",
    "darjeeling": "Darjeeling", "gangtok": "Gangtok", "shillong": "Shillong",
    "meghalaya": "Meghalaya", "kaziranga": "Kaziranga", "rome": "Rome",
    "barcelona": "Barcelona", "amsterdam": "Amsterdam", "prague": "Prague",
    "vienna": "Vienna", "budapest": "Budapest", "istanbul": "Istanbul",
    "cairo": "Cairo", "south africa": "South Africa", "kenya": "Kenya",
    "zanzibar": "Zanzibar", "seychelles": "Seychelles",
    "new york": "New York", "las vegas": "Las Vegas",
    # Hinglish misspellings
    "thalend": "Thailand", "tailand": "Thailand", "thialand": "Thailand",
    "thiland": "Thailand", "thaland": "Thailand",
    "dubei": "Dubai", "kashmeer": "Kashmir", "keshmir": "Kashmir",
    "maldiv": "Maldives", "maldivs": "Maldives",
    "singapor": "Singapore", "singapur": "Singapore",
    "swiss": "Switzerland", "switz": "Switzerland",
    "baly": "Bali", "goaa": "Goa", "keralaa": "Kerala",
    "mauritus": "Mauritius", "vieetnam": "Vietnam",
    # Devanagari
    "थाईलैंड": "Thailand", "थाइलैंड": "Thailand", "थाईलेंड": "Thailand",
    "थाइलेंड": "Thailand",
    "बाली": "Bali", "बली": "Bali",
    "गोवा": "Goa",
    "दुबई": "Dubai",
    "कश्मीर": "Kashmir", "कशमीर": "Kashmir",
    "केरला": "Kerala", "केरल": "Kerala",
    "मालदीव": "Maldives", "मालदीव्स": "Maldives",
    "यूरोप": "Europe",
    "वियतनाम": "Vietnam",
    "सिंगापुर": "Singapore", "सिंगापूर": "Singapore",
    "मनाली": "Manali",
    "शिमला": "Shimla",
    "राजस्थान": "Rajasthan",
    "लद्दाख": "Ladakh",
    "अंडमान": "Andaman",
    "मॉरीशस": "Mauritius",
    "श्रीलंका": "Sri Lanka",
    "नेपाल": "Nepal",
    "भूटान": "Bhutan",
    "पेरिस": "Paris",
    "लंदन": "London",
    "तुर्की": "Turkey",
    "स्विट्जरलैंड": "Switzerland",
    "ग्रीस": "Greece",
    "जापान": "Japan",
    "मलेशिया": "Malaysia",
    "ऑस्ट्रेलिया": "Australia",
    "अमेरिका": "USA",
    "कनाडा": "Canada",
    "जयपुर": "Jaipur",
    "उदयपुर": "Udaipur",
    "ऋषिकेश": "Rishikesh",
    "मुन्नार": "Munnar",
    "ऊटी": "Ooty",
    "नैनीताल": "Nainital",
    "मसूरी": "Mussoorie",
    "आगरा": "Agra",
    "वाराणसी": "Varanasi",
    "अमृतसर": "Amritsar",
    "दार्जिलिंग": "Darjeeling",
    "गंगटोक": "Gangtok",
    "शिलांग": "Shillong",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL WORD BANKS
# ─────────────────────────────────────────────────────────────────────────────
BUDGET_SIGNALS = [
    "lakh", "thousand", "hazaar", "hazar", "hajar",
    "k", "budget", "rupee", "rupaye", "rs", "₹",
    "spend", "kharcha", "cost", "price", "charge",
    "affordable", "cheap", "sasta", "mehnga", "expensive",
    "not sure about price", "pata nahi kitna",
    "30", "40", "50", "60", "70", "80", "90",
]

# Hard no — strong rejection phrases (Devanagari)
# IMPORTANT: bare "नहीं" alone is NOT here — it needs context check
# because "नहीं मैं नहीं कर रहा हूं" has future/present tense that means "not doing it"
# and "नहीं 5 लाख" means budget correction
HARD_NO_DEVANAGARI_PHRASES = [
    "नहीं चाहिए", "interest नहीं", "बिल्कुल नहीं",
    "कोई जरूरत नहीं", "ज़रूरत नहीं", "छोड़ो", "रहने दो", "बंद करो",
    "मत करो", "call मत करो", "मुझे नहीं जाना", "जाना नहीं है",
    "जाना ही नहीं है", "कहीं नहीं जाना", "नहीं जाना है",
    "नहीं कर रहा हूं", "नहीं कर रही हूं", "नहीं करना है",
    "नहीं लेना", "हमें नहीं", "मुझे नहीं",
]

# Devanagari future/soft signals — if present WITH नहीं, it's soft_no not hard no
DEVANAGARI_FUTURE_SOFT = [
    "बाद में", "बाद मे", "जाऊंगा", "जाऊंगी", "जाएंगे", "जायेंगे",
    "करूंगा", "करूंगी", "करेंगे", "सोचेंगे", "सोच लेंगे",
    "देखते हैं", "पता नहीं", "decide नहीं",
]

HARD_NO_REGEX = re.compile(
    r"\b(nahi\s*nahi|bilkul\s*nahi|koi\s*zaroorat\s*nahi|"
    r"zaroorat\s*nahi|rehne\s*do|chhodo|band\s*karo|don'?t\s*call|"
    r"remove|not\s*needed|don'?t\s*want|dont\s*want|no\s*thanks|"
    r"no\s*thank\s*you|never|absolutely\s*not|not\s*at\s*all|"
    r"nahi\s*jaana|nahi\s*chahiye|nahi\s*karna|not\s*interested|"
    r"nahi\s*kar\s*raha|nahi\s*kar\s*rahi|kahin\s*nahi\s*jaana|"
    r"jana\s*hi\s*nahi|jaana\s*hi\s*nahi)\b"
)

SOFT_NO_SIGNALS = [
    "sochenge", "soch lenge", "baad mein dekhenge", "dekhte hain",
    "not sure", "pata nahi", "decide nahi", "abhi nahi socha",
    "wife se poochna", "husband se poochna", "family se poochna",
    "baat karni hai", "discuss karna hai", "thoda time chahiye",
    "kal baat karte hain", "think kar loon", "soch ke batata hoon",
    "soch ke batati hoon", "abhi nahi par", "abhi nahi lekin",
    "baad mein aaunga", "baad mein jaunga",
    # Devanagari soft no
    "बाद में जाऊंगा", "बाद में जाऊंगी", "बाद में जाएंगे",
    "अभी नहीं पर", "अभी नही पर", "बाद में देखेंगे",
    "सोचेंगे", "सोच लेंगे", "पता नहीं अभी",
]

BUSY_DEVANAGARI = [
    "busy", "अभी busy", "बाद में", "थोड़ी देर", "meeting", "office", "driving",
]
BUSY_REGEX = re.compile(
    r"\b(busy|baad\s*mein|baad\s*me|later|meeting|not\s*now|driving|"
    r"occupied|call\s*back|thodi\s*der|abhi\s*free\s*nahi|"
    r"gari\s*chala|office\s*mein|kaam\s*kar\s*raha|kaam\s*kar\s*rahi)\b"
)

# Greeting words — Roman + Devanagari in separate sets
GREETING_ROMAN = {
    "hello", "hi", "hey", "helo", "hii", "heyy", "hlo",
    "namaste", "namaskar", "namasthe", "sat sri akal", "adaab",
    "jai shree krishna", "jai shri ram", "assalamu alaikum",
    "good morning", "good afternoon", "good evening",
    "bol", "bolo", "boliye", "suno", "suniye",
    "okay", "ok", "theek", "theek hai", "acha", "achha",
    # "haan","han","yes","ya","yep","yup" NOT here — they are YES signals
}
GREETING_DEVANAGARI = {
    "हेलो", "हाय", "हे", "नमस्ते", "नमस्कार",
    "बोलो", "बोलिए", "सुनो", "सुनिए", "ओके", "जी",
    # "हाँ","हां","हा" NOT here — they are YES signals
}

FILLER_SET = {
    "umm", "ummm", "hmm", "hmmm", "uh", "uhh", "ah", "ahh", "err",
    "hm", "mm", "mmm",
}

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE VARIATION BANKS
# ─────────────────────────────────────────────────────────────────────────────
DEST_REACTIONS: dict[str, list[str]] = {
    "Goa": [
        "Goa! Wahan bahut maza aata hai, beaches aur khana dono amazing.",
        "Goa bahut accha choice hai! Wahan ka vibe hi alag hota hai.",
        "Goa! Kabhi bhi jaao maza hi maza hai wahan.",
    ],
    "Thailand": [
        "Thailand bahut sundar jagah hai! Beaches bhi, street food bhi — dono kamaal.",
        "Thailand! Wahan bahut log jaate hain — bilkul sahi soch rahe hain aap.",
        "Thailand bahut accha soch rahe hain aap! Popular aur affordable dono hai.",
    ],
    "Bali": [
        "Bali! Wahan ka scene hi alag hota hai — temples, rice fields, sunsets.",
        "Bali bahut sundar jagah hai! Wahan ka peaceful vibe bahut accha lagta hai.",
        "Bali! Romantic aur adventurous dono hai woh jagah.",
    ],
    "Dubai": [
        "Dubai bahut accha hai! Shopping bhi, ghumna bhi — sab kuch milta hai.",
        "Dubai! World class experience milta hai wahan — bahut acchi jagah.",
        "Dubai bahut accha soch rahe hain! Desert safari bhi, malls bhi.",
    ],
    "Kashmir": [
        "Kashmir sach mein bahut sundar jagah hai! India ka jannat kehte hain ise.",
        "Kashmir! Wahan ke pahad aur gardens bahut khoobsurat hain.",
        "Kashmir bahut accha choice hai! Especially abhi ke season mein.",
    ],
    "Maldives": [
        "Maldives! Sapne wali jagah hai yeh — crystal clear paani, white beaches.",
        "Maldives bahut sundar jagah hai! Overwater bungalows toh ekdum amazing hote hain.",
        "Maldives! Ek baar jaao toh yaad rahega hamesha.",
    ],
    "Europe": [
        "Europe! Bahut accha soch rahe hain aap — kaun sa country mainly?",
        "Europe trip! Wahan bahut saari sundar jagahein hain, kaun si jaani hai?",
        "Europe! Paris, Switzerland, Italy — kaun si jagah prefer karenge aap?",
    ],
    "Manali": [
        "Manali bahut pyaari jagah hai! Pahad wahan ke bahut sundar hain.",
        "Manali! Snow aur pahad — bahut accha experience hota hai wahan.",
        "Manali bahut accha choice hai! Wahan ki thandi hawa aur views — kamaal.",
    ],
    "Singapore": [
        "Singapore bahut accha hai! Family ke liye bhi, couple ke liye bhi — perfect.",
        "Singapore! Bahut clean aur organized jagah hai — sab enjoy karte hain.",
        "Singapore bahut accha soch rahe hain! Universal Studios bhi hai wahan.",
    ],
    "Vietnam": [
        "Vietnam bahut sundar jagah hai! Aur relatively sasta bhi — accha deal milta hai.",
        "Vietnam! Bahut underrated jagah hai — jaane wale sab happy hote hain.",
        "Vietnam bahut accha choice hai! Ha Long Bay bahut sundar hai.",
    ],
    "Ladakh": [
        "Ladakh! Wahan ki beauty koi bata nahi sakta, dekhna padta hai.",
        "Ladakh bahut adventurous jagah hai! Mountains aur lakes — out of this world.",
        "Ladakh! Ekdum unique experience milta hai wahan.",
    ],
    "Kerala": [
        "Kerala! God's Own Country kehte hain ise — bahut sundar jagah.",
        "Kerala bahut accha choice hai! Backwaters aur beaches dono ek hi trip mein.",
        "Kerala bahut sundar jagah hai! Wahan ki haryali dil khush kar deti hai.",
    ],
    "Rajasthan": [
        "Rajasthan! Shahi experience milta hai wahan — forts, palaces, culture.",
        "Rajasthan bahut accha choice hai! Jaipur, Udaipur, Jodhpur — sab sundar.",
        "Rajasthan! Indian culture dekhni ho toh yeh jagah best hai.",
    ],
    "Shimla": [
        "Shimla bahut pyaari jagah hai! Colonial charm aur pahad — dono saath.",
        "Shimla! Old world charm aur thandi hawa — bahut popular jagah hai.",
        "Shimla bahut accha choice hai! Mall Road aur pahad — maza aata hai.",
    ],
}
DEFAULT_REACTIONS = [
    "Bahut accha choice hai aapka!",
    "Wahan bahut maza aata hai!",
    "Bahut sundar jagah hai!",
    "Bahut acchi jagah soch rahe hain aap!",
]

# Add London and other European cities to DEST_REACTIONS
DEST_REACTIONS["London"]  = [
    "London bahut sundar sheher hai! History bhi, shopping bhi.",
    "London! Ekdum world class city hai woh.",
    "London bahut accha choice hai — Big Ben, Tower Bridge sab wahan!",
]
DEST_REACTIONS["Paris"]  = [
    "Paris! City of Love — bahut romantic jagah hai.",
    "Paris bahut sundar jagah hai! Eiffel Tower dekhna must hai.",
]
DEST_REACTIONS["Switzerland"] = [
    "Switzerland! Bahut khoobsurat jagah hai — pahad aur snow dono.",
    "Switzerland bahut accha choice hai — ekdum fairy tale jaisi jagah.",
]
DEST_REACTIONS["Japan"] = [
    "Japan bahut unique jagah hai! Culture, food, technology sab amazing.",
    "Japan bahut accha choice hai — cherry blossoms aur temples!",
]
DEST_REACTIONS["Australia"] = [
    "Australia bahut accha hai! Sydney Opera House, beaches — sab kamaal.",
    "Australia bahut sundar jagah hai — bahut diverse experience milta hai.",
]

BUDGET_QUESTIONS = [
    "Budget roughly kitna soch rahe hain aap per person?",
    "Per person kitna laga sakte hain roughly?",
    "Roughly 40-50 hazaar ke aaspaas soch rahe hain, ya thoda alag?",
    "Budget roughly kitna hai aapka — 30-40 hazaar ya thoda zyada?",
    "2 log hain ya family saath hai? Aur budget roughly kitna soch rahe hain?",
]

DEST_QUESTIONS = [
    "Kahan jaana chahte hain aap — koi jagah dimag mein hai?",
    "Koi destination soch rahe hain — Goa, Dubai, Thailand, ya koi aur?",
    "Kahan jaana soch rahe hain — pahad, beach, ya sheher?",
    "Koi plan hai kahan ki? Domestic ya international?",
]

CLOSING_LINES = [
    "Hamare consultant aapko {dest} ke kuch acche options ke saath jald call karenge. Aapka time dene ka shukriya! Achha din rahe aapka!",
    "Hamare travel expert {dest} ke best packages ke saath jald call karenge. Bahut shukriya baat karne ka!",
    "Bahut accha! Hamare consultant {dest} trip ke liye jald call karenge. Take care!",
]

GREETING_RESPONSES_HI = [
    "Hello! Kaise hain aap? Koi trip plan chal rahi hai kya?",
    "Namaste! Bataiye, koi ghumne ka plan hai aajkal?",
    "Hello! Acha laga aapka call! Koi travel plan hai kya?",
    "Haan! Bataiye, koi trip soch rahe hain aap?",
    "Hello! Koi trip ya vacation plan kar rahe hain?",
]
GREETING_RESPONSES_EN = [
    "Hello! How are you? Any trips being planned these days?",
    "Hi there! Good to hear from you! Any travel plans going on?",
    "Hello! Are you planning any trip or vacation?",
    "Hi! Any travel plans on your mind?",
]

SOFT_NO_HI = [
    "Bilkul, sochiye aap! Main bas itna batati hoon — {dest} ke liye {rng} mein bahut acche options hain abhi. Koi sawal ho toh tripinminutes.com pe dekh sakte hain.",
    "Koi baat nahi, no pressure! {dest} ke packages bahut acche chal rahe hain. Jab ready hon, tripinminutes.com pe bhi aa sakte hain.",
]
SOFT_NO_EN = [
    "No worries, take your time! Just so you know, we have great {dest} packages starting {rng}. Feel free to check tripinminutes.com anytime.",
    "Absolutely fine! Great {dest} deals available right now. Visit tripinminutes.com whenever you're ready.",
]


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_pcm16_wav(audio: bytes, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    buf.write(b"RIFF"); buf.write(struct.pack("<I", 36 + len(audio)))
    buf.write(b"WAVE"); buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16)); buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", 1)); buf.write(struct.pack("<I", rate))
    buf.write(struct.pack("<I", rate * 2)); buf.write(struct.pack("<H", 2))
    buf.write(struct.pack("<H", 16))
    buf.write(b"data"); buf.write(struct.pack("<I", len(audio))); buf.write(audio)
    return buf.getvalue()

def pcm16_rms(chunk: bytes) -> float:
    if len(chunk) < 2: return 0.0
    n = len(chunk) // 2
    samples = struct.unpack("<" + "h" * n, chunk[:n * 2])
    return sum(abs(s) for s in samples) / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

def contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))

def clean(text: str) -> str:
    t = (text or "").replace("—", "-").replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()

def sanitize(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"[-•]\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def split_sentences(text: str) -> list[str]:
    t = clean(text)
    if not t: return []
    parts, cur = [], []
    for ch in t:
        cur.append(ch)
        if ch in ".!?।":
            s = "".join(cur).strip()
            if s: parts.append(s[:200])
            cur = []
    if cur:
        r = "".join(cur).strip()
        if r: parts.append(r[:200])
    return [p for p in parts if p] or [t[:200]]

def norm_transcript(text: str) -> str:
    """Normalize STT output. Preserves Devanagari; lowercases Roman."""
    t = clean(text)
    if not t: return ""
    if contains_devanagari(t):
        return clean(t.replace("हूँ", "हूं").replace("हूॅ", "हूं"))
    return t.lower().strip()

def infer_title(text: str, current: str = "") -> str:
    n = text.lower()
    if re.search(r"\b(ma'?am|madam|rahi\s*hoon|rahi\s*hu|kar\s*rahi)\b", n): return "ma'am"
    if re.search(r"\b(sir|raha\s*hoon|raha\s*hu|kar\s*raha)\b", n):          return "sir"
    return current


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
_HINGLISH_MARKERS = {
    "haan","han","nahi","nahin","bhai","yaar","accha","theek",
    "bilkul","chalte","jaana","karna","hota","mein","aap",
    "kya","kab","kahan","kitna","bahut","thoda","zyada",
    "ka","ki","ke","se","ko","ne","par","bhi","hi","toh",
    "lekin","aur","ya","mat","bas","sahi","wahan","woh",
}

def detect_language(text: str) -> str:
    if not text: return "hinglish"
    dev   = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    total = len(text.replace(" ", ""))
    if total == 0: return "hinglish"
    if dev / total > 0.35: return "hindi"
    if re.match(r"^[a-zA-Z0-9\s,\.!?\'\-]+$", text.strip()) and dev == 0:
        if set(text.lower().split()) & _HINGLISH_MARKERS: return "hinglish"
        return "english"
    return "hinglish"


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def is_filler_only(text: str) -> bool:
    words = set(text.lower().strip().split())
    return bool(words) and len(words) <= 2 and words.issubset(FILLER_SET)

def is_greeting_only(text: str) -> bool:
    """
    Works for both Roman AND Devanagari.
    Checks raw tokens (for Devanagari) AND lowercased tokens (for Roman).
    Bug fix: was only checking lowercase, missing हेलो, नमस्ते etc.
    """
    if not text: return False
    raw_words   = set(text.strip().split())
    lower_words = set(text.lower().strip().split())
    combined    = raw_words | lower_words
    if not combined or len(raw_words) > 4: return False
    return all(
        w in GREETING_ROMAN or w in GREETING_DEVANAGARI
        for w in combined
    )

def is_question(text: str) -> bool:
    if "?" in text: return True
    t = text.lower()
    return any(q in t for q in [
        "kya hai","kaisa","kaisi","kaise","kab jaana","kitna lagega",
        "kitne ka","visa chahiye","visa milta","safe hai","best time",
        "weather kaisa","mausam kaisa","flight kitne","hotel kaisa",
        "difference kya","better hai","kaunsa better",
        "company kaisi","trustworthy","genuine","reviews",
        "do i need","is it safe","how much","what is","tell me",
        "bata do","bata sakte","samjhao","batao","explain",
        "kab jayen","kitne din","how many days","how long",
        "mehnga hai","sasta hai","cheap hai","costly",
        "क्या है","कैसा है","कब जाएं","कितना लगेगा","कौनसा बेहतर",
    ])

def has_destination(text: str) -> str:
    """
    Checks longer aliases first (new zealand before new) to avoid partial match.
    Handles English, misspellings, Devanagari.
    """
    if not text: return ""
    t_lower = text.lower().strip()
    for alias, canonical in sorted(DESTINATION_ALIASES.items(), key=lambda x: -len(x[0])):
        if contains_devanagari(alias):
            if alias in text: return canonical
        else:
            if alias in t_lower: return canonical
    return ""

def has_budget(text: str) -> bool:
    t = text.lower()
    # Precise signals only — avoid false positives like "plan" or "trip"
    precise = [
        "lakh","hazaar","hazar","hajar","thousand",
        "rupee","rupaye","rs","₹","kharcha","spend","cost","price",
        "sasta","mehnga","affordable","cheap","expensive","budget",
    ]
    if any(b in t for b in precise): return True
    if any(b in text for b in ["लाख","हज़ार","हजार","₹","रुपये","रुपया"]): return True
    if re.search(r"\b\d{4,6}\b", t): return True  # 4-6 digit number = likely amount
    if re.search(r"\b\d+\s*k\b", t): return True  # "50k" style
    return False

def is_soft_no(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in SOFT_NO_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def classify_intent(text: str) -> str:
    """
    Priority order (most specific first):
    filler → greeting_only → budget_override → no → busy → soft_no → yes → has_budget → unclear

    Key rules:
    - If message has a number/budget word → ALWAYS treat as budget, never as no
    - "नहीं मैं नहीं कर रहा हूं" → no (no budget, no future, just rejection)
    - "नहीं 5 लाख" → has_budget (has number, overrides नहीं)
    - "मेरे को जाना ही नहीं है" → no (explicit कहीं नहीं जाना)
    - "मुझे कहीं नहीं जाना है" → no (explicit rejection)
    - "कोई ऐसा बोल" → unclear (confused, not a no)
    """
    if not text: return "unclear"
    raw  = text.strip()
    norm = raw.lower()

    # 1. Filler — say nothing
    if is_filler_only(norm): return "filler"

    # 2. Pure greeting
    if is_greeting_only(raw): return "greeting_only"

    # 3. BUDGET OVERRIDE — if message has number or money word, it is NEVER a hard no
    #    "नहीं 5 लाख" = customer correcting the amount, not rejecting
    has_num        = bool(re.search(r"\b\d+\b", raw))
    has_money_word = any(w in raw for w in [
        "लाख", "हज़ार", "हजार", "₹", "रुपये", "रुपया",
        "lakh", "hazaar", "hazar", "thousand", "budget",
    ])
    if (has_num or has_money_word) and contains_devanagari(raw):
        return "has_budget"

    # 4. Devanagari-specific checks (order matters — future signals before hard no)
    if contains_devanagari(raw):
        # Future/soft signals WITH नहीं = soft_no not hard no
        if any(p in raw for p in DEVANAGARI_FUTURE_SOFT):
            return "soft_no"
        # Strong rejection phrases
        if any(p in raw for p in HARD_NO_DEVANAGARI_PHRASES):
            return "no"
        # Bare नहीं/नही alone or in sentences like "नहीं कर रहा हूं", "जाना नहीं है"
        # These are hard nos when no budget/future signal present
        if "नहीं" in raw or "नही" in raw:
            # Check if there's a yes/interest signal too
            yes_signals_dev = ("हाँ","हां","बिल्कुल","जाऊंगा","जाऊंगी","जाएंगे","ज़रूर")
            if not any(y in raw for y in yes_signals_dev):
                return "no"
        # Busy
        if any(p in raw for p in BUSY_DEVANAGARI): return "busy"
        # Yes
        if any(p in raw for p in ("हाँ","हां","हा","बिल्कुल","जी हाँ","ठीक है","ठीक")):
            return "yes"

    # 5. Hard no regex (Roman/Hinglish)
    if HARD_NO_REGEX.search(norm):
        # But if budget signal present, it's a correction not a rejection
        if has_budget(text) or re.search(r"\b\d{4,}\b", norm):
            return "has_budget"
        return "no"

    # 6. Single-word hard no
    if re.search(r"^(no|nope|nahi|nahin|nah|bilkul\s*nahi)$", norm.strip()):
        return "no"

    # 7. Busy
    if BUSY_REGEX.search(norm): return "busy"

    # 8. Soft no — hesitation
    if is_soft_no(norm): return "soft_no"

    # 9. Yes signals
    if re.search(
        r"\b(haan|han|yes|ya\b|yap|bilkul|sure|theek\s*hai|interested|"
        r"plan|trip|vacation|chalna|jaana|book|travel|jaata|jaati|jana)\b",
        norm
    ): return "yes"

    # 10. Destination found → treat as yes
    if has_destination(text): return "yes"

    # 11. Budget signal
    if has_budget(text): return "has_budget"

    return "unclear"


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT HEADER — injected into every LLM message
# ─────────────────────────────────────────────────────────────────────────────
def _ctx(obj: Any) -> str:
    parts = []
    if getattr(obj, "customer_title",   ""): parts.append(f"Title:{obj.customer_title}")
    if getattr(obj, "lead_destination", ""): parts.append(f"Destination already given:{obj.lead_destination}")
    if getattr(obj, "lead_budget",      ""): parts.append(f"Budget already given:{obj.lead_budget}")
    lang = getattr(obj, "customer_language", "hinglish")
    parts.append(f"Respond in:{lang}")
    return "[CONTEXT: " + " | ".join(parts) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# VOBIZ XML
# ─────────────────────────────────────────────────────────────────────────────
def build_stream_xml(host: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n'
        f'    <Stream url="wss://{host}/media-stream"\n'
        '            bidirectional="true"\n'
        '            contentType="audio/x-l16;rate=8000"\n'
        '            keepCallAlive="true"/>\n</Response>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# HANGUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
async def parse_payload(request: Request) -> dict:
    out: dict = {}
    try: out.update(dict(await request.form()))
    except Exception: pass
    try:
        body = await request.json()
        if isinstance(body, dict): out.update(body)
    except Exception: pass
    out.update(dict(request.query_params))
    return out

def get_field(payload: dict, *keys: str) -> Any:
    low = {k.lower(): v for k, v in payload.items()}
    for k in keys:
        if k in payload: return payload[k]
        v = low.get(k.lower())
        if v is not None: return v
    return None

def parse_hangup(payload: dict) -> tuple[str, str, str]:
    phone = normalize_phone(get_field(payload, "to", "phone", "callee") or "")
    raw   = str(get_field(payload, "call_status", "status", "hangup_cause") or "").strip()
    dur   = str(get_field(payload, "duration", "bill_duration") or "").strip()
    key   = raw.lower().replace("_", "-")
    if   key == "busy":                          status = "busy"
    elif key in ("no-answer", "timeout"):        status = "no-answer"
    elif key in ("failed", "error", "rejected"): status = "failed"
    elif key in ("completed", "answered"):       status = "completed"
    else:                                        status = "completed" if dur else "failed"
    return phone, status, dur


# ─────────────────────────────────────────────────────────────────────────────
# CALL SESSION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CallSession:
    websocket:   WebSocket
    http_client: httpx.AsyncClient
    llm_client:  Any  = field(default=None)
    llm_model:   str  = field(default="")
    history:     list = field(default_factory=list)
    audio_buf:   bytearray = field(default_factory=bytearray)
    stream_id:   str | None = None
    call_id:     str | None = None

    silence_count:       int   = 0
    greeting_sent:       bool  = False
    opening_in_progress: bool  = False
    is_speaking:         bool  = False
    ignore_until:        float = 0.0

    lock:   asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks:  set          = field(default_factory=set)
    stt_ws: Any          = None
    partial: str         = ""

    stage:             str = "greeting"
    lead_destination:  str = ""
    lead_budget:       str = ""
    customer_title:    str = ""
    customer_language: str = "hinglish"

    used_reactions: set = field(default_factory=set)
    bq_idx:         int = 0
    dq_idx:         int = 0
    last_reply:     str = ""   # prevents exact same reply twice in a row
    unclear_count:  int = 0    # consecutive unclear turns — escalate after 2

    def __post_init__(self):
        if self.llm_client is None: self.llm_client = groq_client
        if not self.llm_model:     self.llm_model   = config.GROQ_MODEL

    def track(self, coro: Any) -> None:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    def _dest_reaction(self, dest: str) -> str:
        """Return destination reaction in the customer's language."""
        lang = self.customer_language

        if lang == "hindi":
            # Devanagari reactions for Hindi customers
            hindi_reactions: dict[str, list[str]] = {
                "Thailand":  ["Thailand bahut sundar jagah hai!", "Thailand! Bahut accha choice hai."],
                "Goa":       ["Goa! Wahan bahut maza aata hai.", "Goa bahut accha choice hai!"],
                "Bali":      ["Bali ka vibe hi alag hota hai!", "Bali! Bahut sundar jagah hai."],
                "Dubai":     ["Dubai bahut accha hai!", "Dubai! World class experience milta hai."],
                "Kashmir":   ["Kashmir sach mein bahut sundar jagah hai!", "Kashmir! Jannat hai apna."],
                "Maldives":  ["Maldives! Sapne wali jagah hai.", "Maldives bahut sundar jagah hai!"],
                "Europe":    ["Europe! Bahut accha soch rahe hain aap.", "Europe trip! Wah."],
                "Manali":    ["Manali bahut pyaari jagah hai!", "Manali! Pahad wahan ke bahut sundar."],
                "Singapore": ["Singapore bahut accha hai!", "Singapore! Family ke liye perfect."],
                "Vietnam":   ["Vietnam bahut sundar jagah hai!", "Vietnam! Bahut acchi jagah hai."],
                "Ladakh":    ["Ladakh! Wahan ki beauty koi bata nahi sakta.", "Ladakh bahut adventurous hai!"],
                "Kerala":    ["Kerala bahut sundar jagah hai!", "Kerala! God's Own Country kehte hain."],
            }
            pool = hindi_reactions.get(dest, [f"{dest} bahut sundar jagah hai!"])

        elif lang == "english":
            english_reactions: dict[str, list[str]] = {
                "Thailand":  ["Thailand is a beautiful choice!", "Great choice — Thailand is amazing!"],
                "Goa":       ["Goa is wonderful!", "Great choice — Goa is always fun!"],
                "Bali":      ["Bali is stunning!", "Bali is a great pick — very peaceful and beautiful."],
                "Dubai":     ["Dubai is an excellent choice!", "Dubai is world-class — great pick!"],
                "Kashmir":   ["Kashmir is breathtaking!", "Great choice — Kashmir is truly paradise."],
                "Maldives":  ["Maldives is a dream destination!", "Maldives — absolutely stunning choice!"],
                "Europe":    ["Europe is wonderful!", "Great thinking — Europe has so much to offer!"],
                "Manali":    ["Manali is beautiful!", "Great pick — Manali is gorgeous!"],
                "Singapore": ["Singapore is a fantastic choice!", "Great pick — Singapore is amazing!"],
                "Vietnam":   ["Vietnam is beautiful and great value!", "Vietnam is a wonderful choice!"],
            }
            pool = english_reactions.get(dest, [f"{dest} is a great choice!"])

        else:
            # Hinglish (default)
            pool = DEST_REACTIONS.get(dest, DEFAULT_REACTIONS)

        unused = [r for r in pool if r not in self.used_reactions]
        if not unused: unused = pool
        pick = random.choice(unused)
        self.used_reactions.add(pick)
        return pick

    def _budget_q(self) -> str:
        # Destination-aware budget range
        dest_ranges = {
            "Goa":"15-30 hazaar","Manali":"15-25 hazaar","Shimla":"15-25 hazaar",
            "Kerala":"25-40 hazaar","Andaman":"30-50 hazaar","Ladakh":"35-55 hazaar",
            "Thailand":"40-70 hazaar","Bali":"45-75 hazaar","Singapore":"50-80 hazaar",
            "Vietnam":"35-60 hazaar","Dubai":"60 hazaar-1 lakh","Nepal":"20-35 hazaar",
            "Maldives":"80 hazaar-1.5 lakh","Europe":"1.5-3 lakh","London":"1-2 lakh",
            "Paris":"1.2-2.5 lakh","Switzerland":"1.5-3 lakh","Australia":"1.5-3 lakh",
            "USA":"1.5-3 lakh","Japan":"80 hazaar-1.5 lakh","Kashmir":"25-45 hazaar",
            "Mauritius":"70 hazaar-1.2 lakh","Rajasthan":"20-35 hazaar",
        }
        lang = self.customer_language
        if self.lead_destination and self.lead_destination in dest_ranges:
            rng  = dest_ranges[self.lead_destination]
            dest = self.lead_destination

            if lang == "english":
                pool = [
                    f"For {dest}, what's your rough budget per person?",
                    f"Most people budget around {rng} for {dest} — does that work for you?",
                    f"What's your approximate budget per person for {dest}?",
                ]
            elif lang == "hindi":
                # Pure Hinglish — no mixed Devanagari+Roman in same sentence
                pool = [
                    f"{dest} ke liye per person roughly kitna budget hai aapka?",
                    f"Per person kitna laga sakte hain roughly {dest} ke liye?",
                    f"Budget roughly kitna hai aapka {dest} trip ke liye?",
                ]
            else:
                # Hinglish
                pool = [
                    f"{dest} ke liye roughly {rng} per person budget soch rahe hain, ya alag?",
                    f"Per person {rng} ke aaspaas soch rahe hain {dest} ke liye?",
                    f"Budget roughly kitna hai aapka — {dest} ke liye {rng} mein accha milta hai.",
                ]
            unused = [q for q in pool if q != self.last_reply]
            q = random.choice(unused if unused else pool)
        else:
            if lang == "english":
                pool = [
                    "What's your rough budget per person?",
                    "Approximately how much are you looking to spend per person?",
                    "What's your budget roughly?",
                ]
            elif lang == "hindi":
                pool = [
                    "Per person roughly kitna budget hai aapka?",
                    "Budget roughly kitna soch rahe hain aap?",
                    "Kitna budget hai aapka per person?",
                ]
            else:
                pool = BUDGET_QUESTIONS
            unused = [q for q in pool if q != self.last_reply]
            q = pool[self.bq_idx % len(pool)] if not unused else random.choice(unused)
        self.bq_idx += 1
        return q

    def _dest_q(self) -> str:
        # Never return same question as last reply
        unused = [q for q in DEST_QUESTIONS if q != self.last_reply]
        if not unused: unused = DEST_QUESTIONS
        q = unused[self.dq_idx % len(unused)]
        self.dq_idx += 1
        return q

    # ── Deepgram STT ──────────────────────────────────────────────────────
    async def start_stt(self) -> None:
        try:
            url = (
                f"wss://api.deepgram.com/v1/listen"
                f"?model={config.STT_MODEL}&language={config.STT_LANGUAGE}"
                f"&encoding=linear16&sample_rate=8000&channels=1"
                f"&punctuate=true&smart_format=true&interim_results=true"
                f"&endpointing={config.STT_ENDPOINTING}"
                f"&utterance_end_ms={config.STT_UTTERANCE_END_MS}"
            )
            ws = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"},
                ping_interval=20,
                ping_timeout=10,
            )
            self.stt_ws = ws
            asyncio.create_task(self._stt_listener())
            logger.info("Deepgram STT connected | endpointing=%sms", config.STT_ENDPOINTING)
        except Exception:
            logger.exception("Deepgram STT failed — batch fallback")
            self.stt_ws = None

    async def _stt_listener(self) -> None:
        try:
            async for msg in self.stt_ws:
                data  = json.loads(msg)
                alts  = data.get("channel", {}).get("alternatives", [])
                if not alts: continue
                text  = alts[0].get("transcript", "").strip()
                final = data.get("is_final", False) or data.get("speech_final", False)
                if text: self.partial = text
                if final and self.partial:
                    full = self.partial
                    self.partial = ""
                    self.track(self.process_speech(full))
        except Exception:
            logger.info("Deepgram STT closed")
        finally:
            self.stt_ws = None

    # ── Vobiz event handler ───────────────────────────────────────────────
    async def handle_event(self, data: dict) -> None:
        event = data.get("event")

        if event == "start":
            sp = data.get("start", {}) if isinstance(data.get("start"), dict) else {}
            self.stream_id = data.get("streamId") or sp.get("streamId")
            self.call_id   = data.get("callId")   or sp.get("callId")
            logger.info("Stream started | stream_id=%s", self.stream_id)
            if not self.greeting_sent:
                self.greeting_sent       = True
                self.opening_in_progress = True
                self.track(self.start_stt())
                self.track(self.send_greeting())
            return

        if event == "media":
            payload = data.get("media", {}).get("payload")
            if not payload: return
            now   = asyncio.get_running_loop().time()
            chunk = base64.b64decode(payload)
            pcm   = audioop.ulaw2lin(chunk, 2)

            # Barge-in: customer speaks while we are talking → stop immediately
            if self.is_speaking:
                rms = pcm16_rms(pcm)
                if rms > BARGE_IN_RMS_THRESHOLD:
                    logger.info("Barge-in (rms=%.0f) — stopping playback", rms)
                    self.is_speaking  = False
                    self.ignore_until = 0.0
                    await self.clear_audio()
                    self.audio_buf.clear()
                    self.silence_count = 0
                return

            if self.opening_in_progress or now < self.ignore_until:
                self.audio_buf.clear(); self.silence_count = 0; return

            if self.stt_ws:
                try:
                    await self.stt_ws.send(pcm)
                except Exception:
                    self.stt_ws = None
                return

            # Batch VAD fallback
            self.audio_buf.extend(chunk)
            rms = pcm16_rms(pcm)
            self.silence_count = self.silence_count + 1 if rms < 500 else 0
            if (self.silence_count >= SILENCE_FRAMES_TO_PROCESS
                    and len(self.audio_buf) >= MIN_SPEECH_BYTES):
                speech = bytes(self.audio_buf)
                self.audio_buf.clear()
                self.silence_count = 0
                self.track(self.process_speech(None, pcm16=audioop.ulaw2lin(speech, 2)))
            return

        if event == "playedStream": logger.info("Checkpoint: %s", data.get("name"))
        if event == "clearedAudio": logger.info("Audio cleared")
        if event == "stop":         logger.info("Stream stopped")

    # ── Speech processing ─────────────────────────────────────────────────
    async def process_speech(self, text: str | None, pcm16: bytes | None = None) -> None:
        async with self.lock:
            if not text and pcm16:
                if len(pcm16) < MIN_SPEECH_BYTES: return
                try:
                    text = await self.batch_stt(pcm16)
                except Exception:
                    logger.exception("Batch STT failed"); return
            if not text: return

            text = norm_transcript(text)

            if is_filler_only(text.lower()):
                logger.info("Filler ignored: %s", text); return

            # Detect language BEFORE build_reply so lang is correct from first message
            # Only update if input is "strong" enough — pure numbers/short words don't change lang
            detected = detect_language(text)
            text_words = len(text.strip().split())
            has_devanagari_in_text = any('ऀ' <= c <= 'ॿ' for c in text)
            is_strong_signal = text_words >= 2 or has_devanagari_in_text
            if is_strong_signal:
                if detected == "hindi":
                    self.customer_language = "hindi"
                elif detected == "english" and self.customer_language != "hindi":
                    # Don't flip from hindi to english on pure English words
                    self.customer_language = "english"
            # Single-word or pure number inputs keep the existing language

            self.customer_title = infer_title(text, self.customer_title)
            logger.info("Customer [%s] lang=%s: %s", self.stage, self.customer_language, text)
            self.history.append({"role": "user", "content": text})

            if self.stage == "done": return

            collected: list[str] = []
            try:
                await self.speak_streamed(
                    self.build_reply(text), collected, checkpoint="reply_done"
                )
            except Exception:
                logger.exception("Reply failed")
                await self.speak("Ek second rukiye.", checkpoint="reply_done")

            full = " ".join(collected).strip()
            if full:
                self.history.append({"role": "assistant", "content": full})
                self.last_reply = full  # track for anti-repeat
            self.history = self.history[-HISTORY_WINDOW:]

    async def batch_stt(self, pcm16: bytes) -> str:
        wav  = build_pcm16_wav(pcm16)
        url  = (f"https://api.deepgram.com/v1/listen"
                f"?model={config.STT_MODEL}&language={config.STT_LANGUAGE}&smart_format=true")
        resp = await self.http_client.post(
            url,
            headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}",
                     "Content-Type": "audio/wav"},
            content=wav, timeout=10.0,
        )
        resp.raise_for_status()
        return str(
            resp.json()
            .get("results", {}).get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        ).strip()

    # ── STATE MACHINE ─────────────────────────────────────────────────────
    async def build_reply(self, text: str):
        intent = classify_intent(text)
        dest   = has_destination(text)
        budget = has_budget(text)
        # Re-check language from current text — handles case where session lang
        # was set on a short previous input (like "haan") and current is Devanagari
        current_detected = detect_language(text)
        if current_detected in ("hindi", "english"):
            lang = current_detected
            self.customer_language = current_detected
        else:
            lang = self.customer_language
        title  = f" {self.customer_title}" if self.customer_title else ""
        is_q   = is_question(text)

        # Track consecutive unclear responses to escalate
        if intent == "unclear":
            self.unclear_count += 1
        else:
            self.unclear_count = 0

        # ── Pure greeting ────────────────────────────────────────────────
        if intent == "greeting_only":
            if self.stage in ("greeting", "got_interest"):
                self.stage = "got_interest"
                yield random.choice(
                    GREETING_RESPONSES_EN if lang == "english" else GREETING_RESPONSES_HI
                )
            elif not self.lead_destination:
                yield self._dest_q()
            else:
                yield self._budget_q()
            return

        # ── Filler — say nothing ─────────────────────────────────────────
        if intent == "filler":
            return

        # ── Hard no ──────────────────────────────────────────────────────
        if intent == "no":
            self.stage = "done"
            if lang == "english":
                yield (
                    f"No problem at all{title}! Whenever you plan a trip, "
                    f"do visit tripinminutes.com — great deals anytime. "
                    f"Have a wonderful day, take care!"
                )
            elif lang == "hindi":
                yield (
                    f"बिल्कुल ठीक है{title}, कोई बात नहीं! "
                    f"जब भी future में trip plan करें — "
                    f"tripinminutes.com पे visit करें। "
                    f"अच्छा दिन रहे आपका!"
                )
            else:
                yield (
                    f"Bilkul theek hai{title}, koi baat nahi! "
                    f"Jab bhi future mein trip plan karein — "
                    f"tripinminutes.com pe visit karein. "
                    f"Achha din rahe aapka, take care!"
                )
            return

        # ── Soft no — hesitation, keep conversation alive ────────────────
        if intent == "soft_no":
            dest_str = self.lead_destination or (
                "popular destinations" if lang == "english" else "popular jagahon"
            )
            rng = "30-50 thousand" if lang == "english" else "30-50 hazaar"
            if lang == "english":
                yield random.choice(SOFT_NO_EN).format(dest=dest_str, rng=rng)
            elif lang == "hindi":
                responses = [
                    f"कोई बात नहीं, आप सोचिए! {dest_str} के लिए {rng} में बहुत अच्छे options हैं। जब ready हों तो tripinminutes.com पे देखें।",
                    f"बिल्कुल ठीक है! कोई जल्दी नहीं। जब भी plan करें, हम available हैं।",
                ]
                yield random.choice(responses)
            else:
                yield random.choice(SOFT_NO_HI).format(dest=dest_str, rng=rng)
            return

        # ── Busy ─────────────────────────────────────────────────────────
        if intent == "busy":
            self.stage = "done"
            if lang == "english":
                yield f"Oh sorry to disturb{title}! You carry on, have a great day!"
            elif lang == "hindi":
                yield f"अरे sorry{title}, disturb कर दिया! आप अपना काम करें, अच्छा दिन रहे!"
            else:
                yield f"Arey sorry{title}, disturb kar diya! Aap apna kaam karein, achha din rahe!"
            return

        # ── Budget correction (e.g. "नहीं 5 लाख") ───────────────────────
        if intent == "has_budget" and self.stage == "got_destination":
            self.lead_budget = text
            self.stage       = "done"
            dest_str = self.lead_destination or ("your trip" if lang == "english" else "aapki trip")
            if lang == "english":
                yield f"Perfect{title}! Our travel consultant will call you with the best {dest_str} options. Thank you, have a great day!"
            elif lang == "hindi":
                closing = random.choice(CLOSING_LINES_HINDI).format(dest=dest_str)
                yield f"बहुत अच्छा{title}! {closing}"
            else:
                closing = random.choice(CLOSING_LINES).format(dest=dest_str)
                yield f"Bahut accha{title}! {closing}"
            return

        # ── Frustrated / confused customer ───────────────────────────────
        # "अरे यार", "kya ho gaya", "ye kya hai" = frustration, not a question
        FRUSTRATION_SIGNALS = [
            "अरे यार", "क्या हो गया", "ye kya hai", "kya ho gaya",
            "kya kar rahe ho", "kya bol rahe", "samajh nahi",
            "समझ नहीं", "क्या हो रहा", "yaar kya", "bhai kya",
        ]
        is_frustrated = any(sig in text.lower() or sig in text for sig in FRUSTRATION_SIGNALS)
        if is_frustrated:
            if lang == "english":
                yield f"Sorry about that{title}! Let me help. Where would you like to go for your trip?"
            elif lang == "hindi":
                yield f"Sorry{title}! Koi baat nahi. Bas bataiye — kahan jaana hai aapko?"
            else:
                yield f"Sorry{title}! Koi confusion nahi — bas bataiye kahan jaana hai aapko?"
            return

        # ── Customer asked a question ────────────────────────────────────
        if is_q and self.stage != "done":
            async for s in self._answer_question(text):
                yield s
            return

        # ── Stage: greeting ──────────────────────────────────────────────
        if self.stage == "greeting":
            if dest:
                self.lead_destination = dest
                self.stage = "got_destination"
                reaction = self._dest_reaction(dest)
                yield f"{reaction} {self._budget_q()}"
            elif intent in ("yes", "has_budget", "unclear"):
                self.stage = "got_interest"
                if lang == "english":
                    yield f"Great{title}! Where are you thinking of going?"
                elif lang == "hindi":
                    yield f"बहुत अच्छा{title}! कहाँ जाना है — Goa, Dubai, Thailand, या कोई और जगह?"
                else:
                    yield f"Bahut accha{title}! {self._dest_q()}"
            else:
                self.stage = "got_interest"
                if lang == "english":
                    yield "No worries! We cover all destinations — domestic and international. Where would you like to go?"
                elif lang == "hindi":
                    yield "कोई बात नहीं! हम सब जगह के लिए trip arrange करते हैं। कहाँ जाना पसंद करेंगे आप?"
                else:
                    yield "Koi baat nahi! Hum sab jagah ke liye trip arrange karte hain. Kahan jaana pasand karenge aap?"
            return

        # ── Stage: got_interest ──────────────────────────────────────────
        if self.stage == "got_interest":
            if dest:
                self.lead_destination = dest
                self.stage = "got_destination"
                reaction = self._dest_reaction(dest)
                yield f"{reaction} {self._budget_q()}"
            elif budget:
                self.lead_budget = text
                if lang == "hindi":
                    yield f"ठीक है{title}! कहाँ जाना है — Goa, Dubai, Thailand, या कोई और जगह?"
                elif lang == "english":
                    yield f"Got it{title}! Where are you planning to go?"
                else:
                    yield f"Theek hai{title}! {self._dest_q()}"
            elif self.unclear_count >= 2:
                # Customer has been unclear 2+ times — change approach completely
                # Don't just repeat destination question — try a different angle
                if lang == "hindi":
                    responses = [
                        "कोई बात नहीं! आप बस बताइए — beach पसंद है, pahad, या koi sheher?",
                        "Domestic trip soch rahe hain ya international?",
                        "Roughly kab jaana soch rahe hain — is saal ya agle saal?",
                    ]
                elif lang == "english":
                    responses = [
                        "No worries! Do you prefer a beach trip, mountains, or a city?",
                        "Are you thinking domestic or international?",
                        "Roughly when are you planning to travel?",
                    ]
                else:
                    responses = [
                        "Koi baat nahi! Beach pasand hai, pahad, ya sheher?",
                        "Domestic soch rahe hain ya international?",
                        "Roughly kab jaana soch rahe hain?",
                    ]
                # Pick one different from last reply
                unused = [r for r in responses if r != self.last_reply]
                yield random.choice(unused if unused else responses)
            else:
                async for s in self._llm_nudge(text, "destination"):
                    yield s
            return

        # ── Stage: got_destination ───────────────────────────────────────
        if self.stage == "got_destination":
            # Explicit Devanagari budget check: "1 लाख", "10 लाख", "50 हज़ार"
            dev_budget = any(b in text for b in ["लाख","हज़ार","हजार","₹","रुपये","रुपया"])
            any_number = bool(re.search(r"\d+", text))  # any digit at all
            if budget or intent == "has_budget" or re.search(r"\b\d{4,}\b", text) or dev_budget or (any_number and dev_budget):
                self.lead_budget = text
                self.stage       = "done"
                dest_str = self.lead_destination or (
                    "your trip" if lang == "english" else "aapki trip"
                )
                if lang == "english":
                    yield (
                        f"Perfect{title}! Our travel consultant will call you shortly "
                        f"with the best {dest_str} options. "
                        f"Thank you for your time! Have a great day!"
                    )
                elif lang == "hindi":
                    closing = random.choice(CLOSING_LINES_HINDI).format(dest=dest_str)
                    yield f"बहुत अच्छा{title}! {closing}"
                else:
                    closing = random.choice(CLOSING_LINES).format(dest=dest_str)
                    yield f"Bahut accha{title}! {closing}"
            else:
                async for s in self._llm_nudge(text, "budget"):
                    yield s
            return

        if self.stage == "done":
            return

        yield self._dest_q()

    # ── LLM: answer question ──────────────────────────────────────────────
    async def _answer_question(self, text: str):
        if not self.lead_destination:
            steer = "After answering, ask which destination they want to go to."
        elif not self.lead_budget:
            steer = f"After answering, ask their budget for {self.lead_destination}."
        else:
            steer = "After answering, say consultant will call soon and say warm bye."

        messages = [
            {"role": "system", "content": SIMRAN_SYSTEM},
            *self.history[-HISTORY_WINDOW:],
            {
                "role": "user",
                "content": (
                    f"{_ctx(self)}\n"
                    f'Customer asked: "{text}"\n'
                    f"Answer in 1 short sentence matching customer language. "
                    f"Then: {steer} "
                    f"Max 2 sentences total. No 'ji' repeated. Use 'aap'."
                ),
            },
        ]
        yielded = 0
        async for s in self._stream_llm(messages):
            cs = sanitize(s)
            if cs:
                yield cs
                yielded += 1
                if yielded >= 2: break
        if yielded == 0:
            # LLM timed out — use direct fallback based on stage
            if self.customer_language == "english":
                if not self.lead_destination:
                    yield "Our consultant will answer all questions. Where would you like to travel?"
                else:
                    yield f"Our consultant will give you all details. What's your rough budget per person?"
            elif self.customer_language == "hindi":
                if not self.lead_destination:
                    yield "Yeh sab hamare consultant batayenge. Kahan jaana chahte hain aap?"
                else:
                    yield f"Yeh sab consultant batayenge. {self._budget_q()}"
            else:
                if not self.lead_destination:
                    yield "Yeh sab hamare consultant batayenge detail mein. Aap kahan jaana chahte hain?"
                else:
                    yield f"Yeh sab consultant batayenge. {self._budget_q()}"

    # ── LLM: nudge ────────────────────────────────────────────────────────
    async def _llm_nudge(self, text: str, missing: str):
        if missing == "destination":
            task = (
                "Customer was unclear about destination. "
                "Ask ONE short warm question to find out where they want to go. "
                "Suggest 2-3 popular examples like Goa, Dubai, Thailand. "
                "Use 'aap'. No 'ji' repeatedly. Max 1 sentence."
            )
        else:
            task = (
                "Customer was unclear about budget. "
                "Ask ONE short casual question about approximate budget. "
                "Give a range like '30-50 hazaar per person'. "
                "Use 'aap'. No 'ji' repeatedly. Max 1 sentence."
            )

        messages = [
            {"role": "system", "content": SIMRAN_SYSTEM},
            *self.history[-HISTORY_WINDOW:],
            {
                "role": "user",
                "content": (
                    f"{_ctx(self)}\n"
                    f'Customer said: "{text}"\n'
                    f"Task: {task}"
                ),
            },
        ]
        yielded = 0
        async for s in self._stream_llm(messages):
            cs = sanitize(s)
            if cs:
                yield cs
                yielded += 1
                break
        if yielded == 0:
            # LLM timed out or failed — use hardcoded fallback (never show error)
            if missing == "destination":
                yield self._dest_q()
            else:
                yield self._budget_q()

    # ── LLM stream ────────────────────────────────────────────────────────
    async def _stream_llm(self, messages: list):
        stream = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=config.GROQ_MAX_TOKENS,
            temperature=config.GROQ_TEMPERATURE,
            stream=True,
        )
        buf = ""
        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            buf  += delta
            while True:
                end = next((i for i, ch in enumerate(buf) if ch in ".!?।"), -1)
                if end == -1: break
                s = buf[:end + 1].strip()
                buf = buf[end + 1:].lstrip()
                if s: yield s
        if buf.strip(): yield buf.strip()

    # ── TTS + playback ────────────────────────────────────────────────────
    async def speak_streamed(self, gen, collected: list[str], *, checkpoint: str | None = None) -> None:
        self.audio_buf.clear(); self.silence_count = 0; self.is_speaking = True
        try:
            await self.clear_audio()
            pending: asyncio.Task | None = None
            async for sentence in gen:
                if not sentence.strip(): continue
                if not self.is_speaking: break   # barge-in
                collected.append(sentence)
                logger.info("Simran: %s", sentence)
                task = asyncio.create_task(self.synthesize(sentence))
                if pending:
                    audio = await pending
                    if audio and self.is_speaking: await self.play(audio)
                pending = task
            if pending and self.is_speaking:
                audio = await pending
                if audio and self.is_speaking: await self.play(audio)
            if checkpoint and self.stream_id:
                await self.websocket.send_text(json.dumps({
                    "event": "checkpoint", "streamId": self.stream_id, "name": checkpoint
                }))
        finally:
            self.is_speaking  = False
            self.ignore_until = asyncio.get_running_loop().time() + POST_PLAYBACK_DELAY
            self.audio_buf.clear(); self.silence_count = 0

    async def speak(self, text: str, checkpoint: str | None = None) -> None:
        self.audio_buf.clear(); self.silence_count = 0; self.is_speaking = True
        try:
            await self.clear_audio()
            for s in split_sentences(text):
                if not self.is_speaking: break
                audio = await self.synthesize(s)
                if audio and self.is_speaking: await self.play(audio)
            if checkpoint and self.stream_id:
                await self.websocket.send_text(json.dumps({
                    "event": "checkpoint", "streamId": self.stream_id, "name": checkpoint
                }))
        finally:
            self.is_speaking  = False
            self.ignore_until = asyncio.get_running_loop().time() + POST_PLAYBACK_DELAY
            self.audio_buf.clear(); self.silence_count = 0

    async def synthesize(self, text: str) -> bytes | None:
        if not config.CARTESIA_API_KEY: return None
        try:
            resp = await self.http_client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "Cartesia-Version": config.CARTESIA_VERSION,
                    "X-API-Key":        config.CARTESIA_API_KEY,
                    "Content-Type":     "application/json",
                },
                json={
                    "model_id":      config.CARTESIA_MODEL,
                    "transcript":    text,
                    "voice":         {"mode": "id", "id": config.CARTESIA_VOICE_ID},
                    "output_format": config.CARTESIA_OUTPUT_FORMAT,
                    "language":      config.CARTESIA_LANGUAGE,
                },
                timeout=12.0,
            )
            resp.raise_for_status()
            return resp.content
        except Exception:
            logger.exception("Cartesia TTS failed"); return None

    async def play(self, audio: bytes, chunk_size: int = 160) -> None:
        for i in range(0, len(audio), chunk_size):
            if not self.is_speaking: break   # honour barge-in inside play loop
            await self.websocket.send_text(json.dumps({
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate":  8000,
                    "payload":     base64.b64encode(audio[i:i + chunk_size]).decode(),
                },
            }))
            await asyncio.sleep(0.02)

    async def clear_audio(self) -> None:
        if not self.stream_id: return
        try:
            await self.websocket.send_text(
                json.dumps({"event": "clearAudio", "streamId": self.stream_id})
            )
        except Exception:
            pass

    async def send_greeting(self) -> None:
        self.history.append({"role": "assistant", "content": GREETING_TEXT})
        try:
            cached = getattr(app.state, "greeting_audio_cache", None)
            if cached is None:
                cached = await self._cache_greeting()
                app.state.greeting_audio_cache = cached
            if cached:
                await self.clear_audio()
                await self.play(cached)
                if self.stream_id:
                    await self.websocket.send_text(json.dumps({
                        "event": "checkpoint", "streamId": self.stream_id, "name": "greeting_done"
                    }))
            else:
                await self.speak(GREETING_TEXT, checkpoint="greeting_done")
        finally:
            self.opening_in_progress = False

    async def _cache_greeting(self) -> bytes | None:
        if not config.CARTESIA_API_KEY: return None
        try:
            resp = await self.http_client.post(
                "https://api.cartesia.ai/tts/bytes",
                headers={
                    "Cartesia-Version": config.CARTESIA_VERSION,
                    "X-API-Key":        config.CARTESIA_API_KEY,
                    "Content-Type":     "application/json",
                },
                json={
                    "model_id":      config.CARTESIA_MODEL,
                    "transcript":    config.fallback_greeting,
                    "voice":         {"mode": "id", "id": config.CARTESIA_VOICE_ID},
                    "output_format": config.CARTESIA_OUTPUT_FORMAT,
                    "language":      config.CARTESIA_LANGUAGE,
                },
                timeout=12.0,
            )
            resp.raise_for_status()
            logger.info("Greeting cached — %d bytes", len(resp.content))
            return resp.content
        except Exception:
            logger.exception("Greeting cache failed"); return None

    async def close(self) -> None:
        if self.stt_ws:
            try: await self.stt_ws.close()
            except Exception: pass
        for t in list(self.tasks): t.cancel()
        if self.tasks: await asyncio.gather(*self.tasks, return_exceptions=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO SUPPORT
# ─────────────────────────────────────────────────────────────────────────────
class _NoOpWS:
    async def send_text(self, _: str) -> None: return

@dataclass
class DemoSession:
    history:           list  = field(default_factory=lambda: [{"role": "assistant", "content": GREETING_TEXT}])
    stage:             str   = "greeting"
    customer_title:    str   = ""
    lead_destination:  str   = ""
    lead_budget:       str   = ""
    customer_language: str   = "hinglish"
    updated_at:        float = field(default_factory=time.time)

class DemoMsgReq(BaseModel):
    session_id: str
    message:    str

def prune_sessions() -> None:
    sessions = getattr(app.state, "demo_sessions", {})
    cutoff   = time.time() - DEMO_SESSION_TTL_SECONDS
    for sid in [k for k, v in sessions.items() if v.updated_at < cutoff]:
        sessions.pop(sid, None)

async def demo_reply(session: DemoSession, text: str) -> tuple[str, str]:
    scratch = CallSession(
        websocket=_NoOpWS(),               # type: ignore
        http_client=app.state.http_client,
        llm_client=groq_client,
        llm_model=config.GROQ_MODEL,
    )
    scratch.history           = session.history.copy()
    scratch.stage             = session.stage
    scratch.customer_title    = session.customer_title
    scratch.lead_destination  = session.lead_destination
    scratch.lead_budget       = session.lead_budget
    scratch.customer_language = session.customer_language

    norm = norm_transcript(text)

    # Always detect from current input — don't rely only on session state
    # Only update language on strong signals (2+ words or has Devanagari)
    detected  = detect_language(norm)
    norm_words = len(norm.strip().split())
    has_dev    = any('ऀ' <= c <= 'ॿ' for c in norm)
    is_strong  = norm_words >= 2 or has_dev
    if is_strong:
        if detected == "hindi":
            scratch.customer_language = "hindi"
            session.customer_language = "hindi"
        elif detected == "english" and session.customer_language != "hindi":
            scratch.customer_language = "english"
            session.customer_language = "english"
    # Short/number inputs keep existing language

    intent = classify_intent(norm)
    if intent == "filler":
        return "Haan, bataiye?", intent

    scratch.customer_title = infer_title(norm, scratch.customer_title)
    scratch.history.append({"role": "user", "content": norm})

    parts: list[str] = []
    async for sentence in scratch.build_reply(norm):
        cs = sanitize(sentence)
        if cs: parts.append(cs)

    reply = " ".join(parts).strip() or (
        "Where would you like to go?" if scratch.customer_language == "english"
        else "Kahan jaana chahte hain aap?"
    )
    scratch.history.append({"role": "assistant", "content": reply})

    session.history          = scratch.history[-HISTORY_WINDOW:]
    session.stage            = scratch.stage
    session.customer_title   = scratch.customer_title
    session.lead_destination = scratch.lead_destination
    session.lead_budget      = scratch.lead_budget
    session.updated_at       = time.time()

    logger.info("Demo | stage=%s dest=%s budget=%s lang=%s | %s",
                session.stage, session.lead_destination,
                session.lead_budget, session.customer_language, reply[:80])
    return reply, intent

async def demo_tts(text: str) -> str | None:
    if not text.strip() or not config.CARTESIA_API_KEY: return None
    t = clean(text)
    if t and t[-1] not in ".!?।": t += "।"
    t = t[:260]
    cache: dict = getattr(app.state, "demo_tts_cache", {})
    if t in cache: return cache[t]
    try:
        resp = await app.state.http_client.post(
            "https://api.cartesia.ai/tts/bytes",
            headers={
                "Cartesia-Version": config.CARTESIA_VERSION,
                "X-API-Key":        config.CARTESIA_API_KEY,
                "Content-Type":     "application/json",
            },
            json={
                "model_id":      config.CARTESIA_MODEL,
                "transcript":    t,
                "voice":         {"mode": "id", "id": config.CARTESIA_VOICE_ID},
                "output_format": config.CARTESIA_TEST_FORMAT,
                "language":      config.CARTESIA_LANGUAGE,
            },
            timeout=12.0,
        )
        if resp.is_error: raise RuntimeError(resp.text[:200])
        encoded = base64.b64encode(resp.content).decode()
        if len(cache) >= 128: cache.pop(next(iter(cache)))
        cache[t] = encoded
        app.state.demo_tts_cache = cache
        return encoded
    except Exception:
        logger.exception("Demo TTS failed"); return None


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    application.state.demo_sessions        = {}
    application.state.demo_tts_cache       = {}
    application.state.greeting_audio_cache = None
    application.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        limits=httpx.Limits(
            max_keepalive_connections=10,
            max_connections=20,
            keepalive_expiry=30.0,
        ),
    )
    logger.info("Simran Telecaller v%s starting on port %s", APP_VERSION, PORT)
    yield
    await application.state.http_client.aclose()

app = FastAPI(
    title="Trip in Minutes — Simran Telecaller",
    version=APP_VERSION,
    lifespan=lifespan,
)

@app.get("/health")
async def health() -> dict:
    return {
        "status":     "ok",
        "version":    APP_VERSION,
        "public_url": PUBLIC_URL,
        "groq":       {"model": config.GROQ_MODEL, "max_tokens": config.GROQ_MAX_TOKENS},
        "cartesia":   {"voice": config.CARTESIA_VOICE_ID, "model": config.CARTESIA_MODEL},
        "deepgram":   {"model": config.STT_MODEL, "language": config.STT_LANGUAGE,
                       "endpointing_ms": config.STT_ENDPOINTING},
        "demo":       f"http://127.0.0.1:{PORT}/demo",
    }

@app.get("/demo", response_class=HTMLResponse)
async def demo_page() -> HTMLResponse:
    if not DEMO_PAGE_PATH.exists():
        raise HTTPException(status_code=500, detail="demo.html missing")
    return HTMLResponse(DEMO_PAGE_PATH.read_text(encoding="utf-8"))

@app.post("/demo/session")
async def demo_session_start() -> dict:
    prune_sessions()
    sid = uuid4().hex
    app.state.demo_sessions[sid] = DemoSession()
    audio = await demo_tts(GREETING_TEXT)
    return {
        "session_id":   sid,
        "greeting":     GREETING_TEXT,
        "audio_base64": audio,
        "audio_mime":   "audio/wav" if audio else None,
        "audio_source": "cartesia" if audio else "browser",
        "model":        config.GROQ_MODEL,
        "message":      "Demo active — no call credits used.",
    }

@app.post("/demo/message")
async def demo_message(payload: DemoMsgReq) -> dict:
    prune_sessions()
    session = app.state.demo_sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    norm = norm_transcript(payload.message)
    if not norm:
        raise HTTPException(status_code=400, detail="Message required.")
    try:
        reply, intent = await asyncio.wait_for(
            demo_reply(session, norm), timeout=DEMO_REPLY_TIMEOUT_SECONDS
        )
    except Exception as e:
        logger.exception("/demo/message failed: %s", e)
        # Never show raw "Sorry, please try again" — give a warm language-aware fallback
        session_lang = getattr(session, "customer_language", "hinglish")
        if session_lang == "hindi":
            reply = "Ek second — koi baat nahi! Kahan jaana chahte hain aap?"
        elif session_lang == "english":
            reply = "One moment please! Where are you planning to travel?"
        else:
            reply = "Ek second rukiye! Kahan jaana chahte hain aap?"
        intent = "error"
    try:
        audio = await asyncio.wait_for(demo_tts(reply), timeout=DEMO_TTS_TIMEOUT_SECONDS)
    except Exception:
        audio = None
    return {
        "reply":           reply,
        "audio_base64":    audio,
        "audio_mime":      "audio/wav" if audio else None,
        "intent":          intent,
        "stage":           session.stage,
        "destination":     session.lead_destination,
        "budget":          session.lead_budget,
        "done":            session.stage == "done",
        "model":           config.GROQ_MODEL,
        "display_message": clean(payload.message),
    }

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def incoming_call(request: Request) -> Response:
    host = request.headers.get("host", "")
    logger.info("Incoming call → wss://%s/media-stream", host)
    return Response(content=build_stream_xml(host), media_type="text/xml")

@app.api_route("/hangup", methods=["GET", "POST"])
async def hangup(request: Request) -> dict:
    try:
        payload = await parse_payload(request)
        phone, status, duration = parse_hangup(payload)
        if phone:
            update_status(phone, status, outcome=status, duration=duration)
            logger.info("Lead %s → %s (%ss)", phone, status, duration)
    except Exception:
        logger.exception("Hangup error")
    return {"status": "ok"}

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    session = CallSession(websocket=ws, http_client=app.state.http_client)
    try:
        while True:
            data = json.loads(await ws.receive_text())
            await session.handle_event(data)
            if data.get("event") == "stop": break
    except WebSocketDisconnect:
        logger.info("Vobiz disconnected | dest=%s budget=%s",
                    session.lead_destination, session.lead_budget)
    except Exception:
        logger.exception("WebSocket error")
    finally:
        await session.close()
        try: await ws.close()
        except Exception: pass


if __name__ == "__main__":
    print("=" * 60)
    print("  Trip in Minutes — Simran Telecaller v7.0")
    print("  Goal : Destination + budget in 30-40 sec, warm close")
    print("  Style: Simple Hinglish, respectful, real human feel")
    print()
    print("  Terminal 1 : cloudflared tunnel --url http://localhost:5050")
    print("  Terminal 2 : python main.py")
    print("  Terminal 3 : python dashboard.py")
    print("  Terminal 4 : python caller.py --limit 1")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)