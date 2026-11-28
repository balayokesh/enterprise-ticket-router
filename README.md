# Automated Support Ticket Triage & Analytics Pipeline

An asynchronous Python pipeline that classifies and enriches support tickets via OpenRouter, grounds recommended replies in internal runbooks using pgvector + Gemini embeddings (RAG), logs LLMOps telemetry separately from business outputs, and provides a Flask web dashboard with a Human-in-the-Loop (HITL) review workflow.

---

## Tech Stack
- **Language & Web:** Python, Flask
- **Data & Validation:** Pandas, Pydantic (JSON Structured Outputs)
- **LLM Gateway & Model:** OpenRouter API (`meta-llama/llama-3.3-70b-instruct`)
- **RAG & Vector Store:** PostgreSQL + `pgvector` (Docker), Google Gemini Embeddings (`gemini-embedding-001`)

---

## Local Setup & Installation

1. **Prerequisites:**
   - Python 3.10+
   - Docker & Docker Compose (for the local `pgvector` database)

2. **Clone and switch to `dev`:**
   ```bash
   git clone https://github.com/balayokesh/enterprise-ticket-router.git
   cd enterprise-ticket-router
   ```

3. **Create a virtual environment and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy `.env.example` to `.env` and fill in your API keys (note that `docker-compose.yml` maps Postgres to host port `5433`):
   ```env
   OPENROUTER_API_KEY="your-openrouter-api-key-here"
   GEMINI_API_KEY="your-gemini-api-key-here"
   DATABASE_URL="postgresql://postgres:postgres@localhost:5433/ticket_rag"
   EMBEDDING_MODEL="gemini-embedding-001"
   ```

5. **Start the `pgvector` database and ingest the knowledge base:**
   ```bash
   docker compose up -d
   python rag_ingest.py
   ```

---

## Usage

### Run CLI Pipeline
```bash
python main.py
```
Outputs `enriched_data.csv` (triage, `Suggested_Reply`, `Cited_Sources`, `Review_Status`) and `telemetry_metrics.csv` (token usage, cost, LLM & retrieval latency).

### Run Web Dashboard (with HITL Review)
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser to filter tickets, upload CSV batches, inspect RAG-cited runbooks, and approve or edit replies before sending.
