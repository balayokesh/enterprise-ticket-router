import os
import re
import json
import asyncio
import time
from typing import Literal

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag_retriever import retrieve_relevant_runbooks

load_dotenv()

MODEL = "meta-llama/llama-3.3-70b-instruct"

# Approximate OpenRouter pricing for Llama 3.3 70B (per 1M tokens)
COST_PER_PROMPT_TOKEN = 0.70 / 1_000_000
COST_PER_COMPLETION_TOKEN = 0.90 / 1_000_000

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class TicketTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ai_urgency: Literal["High", "Medium", "Low"] = Field(description="Urgency level")
    ai_category: Literal["Billing", "Technical", "Hardware", "Account", "Security"] = Field(description="Ticket category")
    requires_immediate_escalation: bool = Field(
        description="True if it needs human intervention, False if it can be auto-resolved"
    )
    summary_sentiment: Literal["Frustrated", "Neutral", "Satisfied"] = Field(description="Customer sentiment")
    suggested_reply: str = Field(
        description=(
            "Concise, empathetic customer-facing draft reply strictly grounded in the retrieved "
            "runbook sections below. If no runbook sections apply, provide a polite holding message."
        )
    )
    cited_sources: list[str] = Field(
        description=(
            "Exact citation strings (e.g. 'GoPro_Hero_Runbook.pdf § Charging Issues') used from "
            "the retrieved chunks. Empty list if no runbook was used."
        )
    )

def clean_text(value, default="") -> str:
    """Return a safe string for prompt use (handles NaN / None)."""
    if value is None or pd.isna(value):
        return default
    return str(value).strip() or default

def parse_llm_json(raw: str) -> dict:
    """Tolerant JSON parser: strips code fences, extracts the {...} block,
    and quotes bare words such as  "ai_urgency": Medium."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = re.sub(
            r'(:\s*)(?!true\b|false\b|null\b|-?\d)([A-Za-z][A-Za-z_ ]*?)(\s*[,}\n])',
            r'\1"\2"\3',
            raw,
        )
        return json.loads(repaired)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,))
)
async def async_triage_ticket(prompt: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        req_start = time.perf_counter()
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600,
        )
        req_latency = time.perf_counter() - req_start

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("Response was truncated (finish_reason='length')")
        content = choice.message.content
        if not content:
            raise ValueError("Empty response from model")

        data = parse_llm_json(content)

        for key in ("ai_urgency", "ai_category", "summary_sentiment"):
            if isinstance(data.get(key), str):
                data[key] = data[key].strip().capitalize()

        # Ensure cited_sources is always a list
        if not isinstance(data.get("cited_sources"), list):
            data["cited_sources"] = []

        triage_obj = TicketTriage.model_validate(data)

        usage = response.usage
        telemetry = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "latency_sec": round(req_latency, 3),
        }
        return triage_obj, telemetry

def build_prompt(product: str, subject: str, description: str, runbook_chunks: list[dict]) -> str:
    if runbook_chunks:
        runbook_block = "\n\n".join(
            f"[{c['citation']}]\n{c['content']}" for c in runbook_chunks
        )
        runbook_section = (
            "RETRIEVED RUNBOOK SECTIONS (ground all troubleshooting steps and policies strictly in these):\n"
            f"{runbook_block}"
        )
    else:
        runbook_section = (
            "RETRIEVED RUNBOOK SECTIONS: None available. "
            "If no runbook applies, the suggested_reply must be a polite holding message."
        )

    return f"""
    Analyze the following customer support ticket and return a VALID JSON object.
    CRITICAL: Every string value must be in double quotes, and boolean must be lowercase true/false.

    Use EXACTLY these six keys:
    - "ai_urgency": "High", "Medium", or "Low"
    - "ai_category": "Billing", "Technical", "Hardware", "Account", or "Security"
    - "requires_immediate_escalation": true or false
    - "summary_sentiment": "Frustrated", "Neutral", or "Satisfied"
    - "suggested_reply": A concise, empathetic customer-facing reply strictly grounded in the runbook sections above (or a polite holding message if none apply). Do NOT invent policies.
    - "cited_sources": A JSON array of the exact citation strings you used (e.g. ["GoPro_Hero_Runbook.pdf § Charging Issues"]). Use an empty array [] if no runbook was used.

    {runbook_section}

    Product: {product}
    Subject: {subject}
    Description: {description}
    """

async def process_single_ticket(index, row, semaphore):
    ticket_id = row.get("Ticket ID", index)
    product = clean_text(row.get("Product Purchased", row.get("ProductPurchased")), "Unknown Product")
    subject = clean_text(row.get("Ticket Subject"), "General Inquiry")
    description = clean_text(row.get("Ticket Description"), "")

    # --- RAG retrieval (offloaded to thread to keep the event loop free) ---
    runbook_chunks, retrieval_latency_ms = await asyncio.to_thread(
        retrieve_relevant_runbooks,
        product, subject, description,
        3,   # top_k
    )

    prompt = build_prompt(product, subject, description, runbook_chunks)

    try:
        triage, telemetry = await async_triage_ticket(prompt, semaphore)
        cost = (telemetry["prompt_tokens"] * COST_PER_PROMPT_TOKEN) + (
            telemetry["completion_tokens"] * COST_PER_COMPLETION_TOKEN
        )
        cited = " | ".join(triage.cited_sources) if triage.cited_sources else "None"
        return index, {
            "business": {
                "AI_Urgency": triage.ai_urgency,
                "AI_Category": triage.ai_category,
                "AI_Escalate": triage.requires_immediate_escalation,
                "AI_Sentiment": triage.summary_sentiment,
                "Suggested_Reply": triage.suggested_reply,
                "Cited_Sources": cited,
                "Review_Status": "Pending Review",
            },
            "telemetry": {
                "Ticket ID": ticket_id,
                "Model": MODEL,
                "Status": "Success",
                "Latency_Sec": telemetry["latency_sec"],
                "Prompt_Tokens": telemetry["prompt_tokens"],
                "Completion_Tokens": telemetry["completion_tokens"],
                "Total_Tokens": telemetry["total_tokens"],
                "Estimated_Cost_USD": round(cost, 6),
                "Retrieval_Latency_Ms": retrieval_latency_ms,
                "Retrieved_Chunks": len(runbook_chunks),
            },
        }
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, {
            "business": {
                "AI_Urgency": "Error",
                "AI_Category": "Error",
                "AI_Escalate": False,
                "AI_Sentiment": "Error",
                "Suggested_Reply": "",
                "Cited_Sources": "None",
                "Review_Status": "Pending Review",
            },
            "telemetry": {
                "Ticket ID": ticket_id,
                "Model": MODEL,
                "Status": f"Failed: {type(e).__name__}",
                "Latency_Sec": 0.0,
                "Prompt_Tokens": 0,
                "Completion_Tokens": 0,
                "Total_Tokens": 0,
                "Estimated_Cost_USD": 0.0,
                "Retrieval_Latency_Ms": retrieval_latency_ms,
                "Retrieved_Chunks": len(runbook_chunks),
            },
        }

async def async_enrich_pipeline(
    input_path="clean_data.csv",
    output_path="enriched_data.csv",
    telemetry_path="telemetry_metrics.csv",
    limit=5,
):
    print("Loading cleaned dataset...")
    df = pd.read_csv(input_path)
    subset = df.head(limit).copy()

    semaphore = asyncio.Semaphore(5)
    print(f"Processing {len(subset)} tickets asynchronously with decoupled telemetry...")

    wall_start = time.perf_counter()
    tasks = [process_single_ticket(index, row, semaphore) for index, row in subset.iterrows()]
    results = await asyncio.gather(*tasks)
    wall_duration = time.perf_counter() - wall_start

    sorted_results = [res for _, res in sorted(results, key=lambda x: x[0])]

    # 1. Populate clean business output DataFrame
    subset["AI_Urgency"] = [r["business"]["AI_Urgency"] for r in sorted_results]
    subset["AI_Category"] = [r["business"]["AI_Category"] for r in sorted_results]
    subset["AI_Escalate"] = [r["business"]["AI_Escalate"] for r in sorted_results]
    subset["AI_Sentiment"] = [r["business"]["AI_Sentiment"] for r in sorted_results]
    subset["Suggested_Reply"] = [r["business"]["Suggested_Reply"] for r in sorted_results]
    subset["Cited_Sources"] = [r["business"]["Cited_Sources"] for r in sorted_results]
    # Default Review_Status to "Pending Review"; overwrite only the rows just processed
    if "Review_Status" not in subset.columns:
        subset["Review_Status"] = "Pending Review"
    else:
        subset["Review_Status"] = subset["Review_Status"].fillna("Pending Review")
    subset["Review_Status"] = [r["business"]["Review_Status"] for r in sorted_results]
    subset.to_csv(output_path, index=False)

    # 2. Populate separate operational telemetry DataFrame
    telemetry_df = pd.DataFrame([r["telemetry"] for r in sorted_results])
    telemetry_df.to_csv(telemetry_path, index=False)

    # 3. Log aggregate summary
    total_prompt = telemetry_df["Prompt_Tokens"].sum()
    total_completion = telemetry_df["Completion_Tokens"].sum()
    total_tokens = telemetry_df["Total_Tokens"].sum()
    total_cost = telemetry_df["Estimated_Cost_USD"].sum()

    print("\n" + "=" * 52)
    print("PIPELINE TELEMETRY & PERFORMANCE SUMMARY")
    print("=" * 52)
    print(f"• Tickets Processed       : {len(subset)}")
    print(f"• Wall-Clock Duration     : {wall_duration:.2f} s")
    print(f"• Avg Latency / Request   : {telemetry_df['Latency_Sec'].mean():.2f} s")
    print(f"• Total Prompt Tokens     : {total_prompt:,}")
    print(f"• Total Completion Tokens : {total_completion:,}")
    print(f"• Grand Total Tokens      : {total_tokens:,}")
    print(f"• Estimated Total Cost    : ${total_cost:.6f}")
    print("=" * 52)
    print(f"Saved clean business data to : {output_path}")
    print(f"Saved operational metrics to : {telemetry_path}")

def enrich_support_tickets(
    input_path="clean_data.csv",
    output_path="enriched_data.csv",
    telemetry_path="telemetry_metrics.csv",
    limit=5,
):
    asyncio.run(async_enrich_pipeline(input_path, output_path, telemetry_path, limit))

if __name__ == "__main__":
    enrich_support_tickets()
