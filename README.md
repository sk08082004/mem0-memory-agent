# Mem0 Long-Term Memory Agent

A long-term memory AI agent built with **Python, Google Gemini, and Mem0** for persistent context across conversations.

## Overview

Traditional LLMs are stateless — once a conversation ends, context is lost. This project adds a memory layer around Gemini that decides, extracts, stores, retrieves, and updates information for future conversations.

## Tech Stack

- Python
- Google Gemini (response generation + memory decisions)
- Mem0 (long-term semantic memory)
- Python-dotenv
- CLI

## How It Works

1. User sends a message
2. Relevant memories are retrieved from Mem0
3. Gemini generates a response
4. Gemini decides if anything is worth remembering
5. If yes, key info is extracted and stored/updated in Mem0

## Features

- Short-term conversation context (rolling window)
- Long-term semantic memory via Mem0
- Intelligent memory decision (no hardcoded rules)
- Memory extraction from large messages
- Memory updating (recency-based conflict resolution)
- Persists across restarts
- Graceful error handling

## CLI Commands

| Command | Description |
|---|---|
| `/memories` | Show stored memories |
| `/forget <n>` | Delete a specific memory |
| `/clear` | Delete all memories |
| `/new` | Start new chat (keeps long-term memory) |
| `/help` | Show commands |
| `exit` | Quit |

## Project Structure

mem0-memory-agent/
├── app/
│   ├── main.py
│   ├── agent.py
│   ├── memory.py
│   └── config.py
├── data/
├── .env
└── requirements.txt



## Setup

```bash
git clone <YOUR_REPOSITORY_URL>
cd mem0-memory-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:


Run:

```bash
python -m app.main
```

## Status

Core agent, memory pipeline, and CLI are complete. Web UI is in progress.

## Author

**Sunny Kumar**