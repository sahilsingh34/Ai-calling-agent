# Simran AI Telecaller Agent (v7.0)

Simran is a high-performance, voice-driven AI telecaller agent designed for **Trip in Minutes**. She acts as a warm, professional travel consultant who specializes in collecting trip details (destination and budget) from customers through natural voice conversations.

![Project Version](https://img.shields.io/badge/version-7.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-green)
![Framework](https://img.shields.io/badge/framework-FastAPI-009688)

## 🚀 Key Features

- **Natural Voice Conversations**: Powered by Cartesia's life-like Hindi/Indian-English voices.
- **Bilingual Intelligence**: Seamlessly handles Hindi, English, and Hinglish (code-switching) using Deepgram's multi-language STT.
- **Latency Optimized**: Engineered for sub-second response times (FastAPI + Groq Llama 3) to keep conversations fluid and human-like.
- **Advanced Intent Detection**: Custom logic to identify destinations, budgets, hesitations (soft-nos), rejections (hard-nos), and status (busy).
- **Smart Barge-in**: Simran stops speaking immediately when the customer starts talking, mimicking real human behavior.
- **Web Demo Interface**: Includes a browser-based demo page for testing the agent's intelligence and voice without telephony.

## 🛠️ Technology Stack

- **LLM**: [Groq](https://groq.com/) (Llama 3.3 70B) for rapid language processing.
- **STT**: [Deepgram](https://deepgram.com/) for high-accuracy speech-to-text with multi-language support.
- **TTS**: [Cartesia](https://cartesia.ai/) for high-fidelity, low-latency text-to-speech.
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) for a robust, asynchronous web server.
- **Telephony**: [Vobiz](https://vobiz.com/) integration for handling real voice calls.
- **Database**: SQLite for lead management and training data storage.

## 📁 Project Structure

- `main.py`: Core application logic, FastAPI routes, and telephony websocket handling.
- `config.py`: Centralized configuration and environment variable management.
- `database.py`: SQLite database schema and helper functions for lead tracking.
- `intent_engine.py`: Specialized logic for classifying user intents and extracting data.
- `prompt_engine.py`: Manages the personality and instructions for the LLM.
- `demo.html`: Frontend interface for browser-based testing.
- `.env.example`: Template for required API keys and configuration.

## ⚙️ Setup & Installation

### 1. Prerequisite
- Python 3.9 or higher.
- A virtual environment is highly recommended.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
Copy the `.env.example` file to a new file named `.env` and fill in your API keys:
```bash
cp .env.example .env
```
Key requirements:
- `GROQ_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`

### 4. Running the Project
Start the FastAPI server:
```bash
python main.py
```
The server will start on the port specified in your `.env` (default: 5050).

## 🛡️ Security Note

This project uses environment variables (`.env`) to store sensitive API keys. **Never commit your `.env` file to version control.** A `.gitignore` has been included to ensure secrets stay local.

## 📞 Usage

- **Telephony**: Configure your Vobiz webhook to point to `https://your-domain/voice`.
- **Demo**: Access the interactive demo at `http://localhost:5050/demo`.

---
*Developed for Trip in Minutes — "Plan your trip in minutes."*
