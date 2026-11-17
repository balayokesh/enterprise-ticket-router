import os
import re
import json
import asyncio
from typing import Literal

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

MODEL = "meta-llama/llama-3.3-70b-instruct"

# Initialize the Async OpenRouter client
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

def extract_content(response) -> str:
    """Pull the message text out of the response and sanity-check it."""
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("Response was truncated (finish_reason='length')")
    content = choice.message.content
    if not content:
        raise ValueError("Empty response from model")
    return content

# Retry decorator for handling rate limits (429) or temporary server errors
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,))
)
async def async_triage_ticket(prompt: str, semaphore: asyncio.Semaphore) -> TicketTriage:
    async with semaphore:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300,
        )
        raw_content = extract_content(response)
        data = parse_llm_json(raw_content)

    # Normalize casing for string fields (e.g. "medium" -> "Medium")
    for key in ("ai_urgency", "ai_category", "summary_sentiment"):
        if isinstance(data.get(key), str):
            data[key] = data[key].strip().capitalize()

        return TicketTriage.model_validate(data)

def build_prompt(product: str, subject: str, description: str) -> str:
    return f"""
    Analyze the following customer support ticket and return a VALID JSON object.
    CRITICAL: Every string value must be in double quotes, and boolean must be lowercase with NO quotes (true/false).

    Use EXACTLY these four keys:
    - "ai_urgency": "High", "Medium", or "Low"
    - "ai_category": "Billing", "Technical", "Hardware", "Account", or "Security"
    - "requires_immediate_escalation": true or false (boolean, not a string)
    - "summary_sentiment": "Frustrated", "Neutral", or "Satisfied"

    Product: {product}
    Subject: {subject}
    Description: {description}
    """

async def process_single_ticket(index, row, semaphore):
    product = clean_text(row.get("Product Purchased", row.get("ProductPurchased")), "Unknown Product")
    subject = clean_text(row.get("Ticket Subject"), "General Inquiry")
    description = clean_text(row.get("Ticket Description"), "")
    prompt = build_prompt(product, subject, description)

    try:
        triage = await async_triage_ticket(prompt, semaphore)
        return index, {
            "AI_Urgency": triage.ai_urgency,
            "AI_Category": triage.ai_category,
            "AI_Escalate": triage.requires_immediate_escalation,
            "AI_Sentiment": triage.summary_sentiment,
        }
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, {
            "AI_Urgency": "Error",
            "AI_Category": "Error",
            "AI_Escalate": False,
            "AI_Sentiment": "Error",
        }

async def async_enrich_pipeline(input_path="clean_data.csv", output_path="enriched_data.csv", limit=5):
    print("Loading cleaned dataset...")
    df = pd.read_csv(input_path)
    subset = df.head(limit).copy()

    # Limit concurrency to 5 simultaneous requests to respect rate limits
    semaphore = asyncio.Semaphore(5)

    print(f"Processing {len(subset)} tickets asynchronously via OpenRouter...")
    tasks = [
        process_single_ticket(index, row, semaphore)
        for index, row in subset.iterrows()
    ]

    results = await asyncio.gather(*tasks)

    # Map results back to dataframe
    urgencies, categories, escalations, sentiments = [], [], [], []
    for _, res in results:
        urgencies.append(res["AI_Urgency"])
        categories.append(res["AI_Category"])
        escalations.append(res["AI_Escalate"])
        sentiments.append(res["AI_Sentiment"])

    subset["AI_Urgency"] = urgencies
    subset["AI_Category"] = categories
    subset["AI_Escalate"] = escalations
    subset["AI_Sentiment"] = sentiments

    subset.to_csv(output_path, index=False)
    print(f"AI enrichment complete! Saved output to {output_path}")
    print(subset[["Ticket ID", "AI_Urgency", "AI_Category", "AI_Escalate", "AI_Sentiment"]])

def enrich_support_tickets(input_path="clean_data.csv", output_path="enriched_data.csv", limit=5):
    asyncio.run(async_enrich_pipeline(input_path, output_path, limit))

if __name__ == "__main__":
    enrich_support_tickets()
