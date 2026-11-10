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
   Set your OpenRouter API key and the OpenAI base URL so the client routes requests through OpenRouter properly.

   * **Windows (PowerShell):**
     ```powershell
     $env:OPENROUTER_API_KEY="your-openrouter-api-key-here"
     ```

   * **Mac / Linux (Bash / Zsh):**
     ```bash
     export OPENROUTER_API_KEY="your-openrouter-api-key-here"
     ```

---

## Running the Pipeline in Order

Execute the pipeline modules sequentially to process your tickets from raw data to final analytics:
1. Clean the input csv file (data.csv -> clean_data.csv)
   ```bash
   python pipeline.py
   ```
2. AI Enrichment & Triage
   Processes the support records through the LLM and outputs `enriched_data.csv`.
   ```bash
   python ai_enrichment.py
   ```  
3. Generate analytics/report from the `enriched_data.csv`
   ```bash
   python reporting.py
   ```
