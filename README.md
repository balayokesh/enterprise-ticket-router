# Automated Support Ticket Triage & Analytics Pipeline

An asynchronous Python pipeline that classifies and enriches support tickets via OpenRouter, logs LLMOps telemetry separately from business outputs, and provides a Flask web dashboard for exploration and batch runs.

---

## Tech Stack
- **Language & Web:** Python, Flask
- **Data & Validation:** Pandas, Pydantic (JSON Structured Outputs)
- **LLM Gateway & Model:** OpenRouter API (`meta-llama/llama-3.3-70b-instruct`)

---

## Setup & Installation

1. **Clone and switch to `dev`:**
   ```bash
   git clone [https://github.com/balayokesh/enterprise-ticket-router.git](https://github.com/balayokesh/enterprise-ticket-router.git)
   cd enterprise-ticket-router
   git checkout dev
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file from `.env.example` and add your API key:
   ```env
   OPENROUTER_API_KEY="your-openrouter-api-key-here"
   ```

---

## Usage

### Run CLI Pipeline
```bash
python main.py
```
(or)
### Run via Web Browser
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.
