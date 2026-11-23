# Automated Support Ticket Triage & Analytics Pipeline

A robust, production-grade Python data pipeline that ingests customer support tickets, enriches them using Large Language Models via OpenRouter, and generates executive reporting metrics.

---


## Tech Stack & Tools
* **Language:** Python
* **Data Processing:** Pandas
* **LLM Gateway:** OpenRouter API (`https://openrouter.ai/api/v1`)
* **Model:** Meta Llama 3.3 70B Instruct
* **Validation & Formatting:** Pydantic, JSON Structured Outputs

---

## Setup & Installation

1. **Clone the repository & switch to the development branch:**
   ```bash
   git clone https://github.com/balayokesh/enterprise-ticket-router.git
   cd support-ticket-pipeline
   git checkout dev
   ```

2. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**  
   Copy the `.env.example` template to create a `.env` file in the root directory, then add your OpenRouter API key and base URL:
   ```env
   OPENROUTER_API_KEY="your-openrouter-api-key-here"
   ```
   
---

## Running the Pipeline

Execute the script to process your tickets from raw data to final analytics:
   ```bash
   python main.py
   ```

## Architecture & Telemetry
**Operational Telemetry (`telemetry_metrics.csv will be generated after execution`)**: Tracks LLMOps metrics per ticket, including API latency, prompt/completion token usage, model version, and estimated inference cost.
