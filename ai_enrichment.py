import os
import re
import json
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

MODEL = "meta-llama/llama-3.3-70b-instruct"

# Initialize the OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class TicketTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")  # needed for strict JSON schema mode

    ai_urgency: Literal["High", "Medium", "Low"] = Field(
        description="Urgency level: High, Medium, or Low"
    )
    ai_category: Literal["Billing", "Technical", "Hardware", "Account", "Security"] = Field(
        description="Ticket category"
    )
    requires_immediate_escalation: bool = Field(
        description="True if it needs human intervention, False if it can be auto-resolved"
    )
    summary_sentiment: Literal["Frustrated", "Neutral", "Satisfied"] = Field(
        description="Customer sentiment"
    )

def clean_text(value, default=""):
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
        raise ValueError("Response was truncated (finish_reason='length'); raise max_tokens")
    content = choice.message.content
    if not content:
        raise ValueError("Empty response from model")
    return content

def triage_ticket(prompt: str) -> TicketTriage:
    # Plain json_object mode with tolerant parsing
    response = client.chat.completions.create(
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

    # Pydantic coerces "false"/"true" strings to real booleans
    return TicketTriage.model_validate(data)

def build_prompt(product: str, subject: str, description: str) -> str:
    return f"""
    Analyze the following customer support ticket and return a VALID JSON object.
    CRITICAL: Every string value must be in double quotes (e.g., "Medium", not Medium),
    and the boolean must be lowercase true or false with NO quotes.

    Use EXACTLY these four keys:
    - "ai_urgency": "High", "Medium", or "Low"
    - "ai_category": "Billing", "Technical", "Hardware", "Account", or "Security"
    - "requires_immediate_escalation": true or false (boolean, not a string)
    - "summary_sentiment": "Frustrated", "Neutral", or "Satisfied"

    Product: {product}
    Subject: {subject}
    Description: {description}
    """

def enrich_support_tickets(input_path="clean_data.csv", output_path="enriched_data.csv", limit=5):
    print("Loading cleaned dataset...")
    df = pd.read_csv(input_path)
    subset = df.head(limit).copy()

    urgencies, categories, escalations, sentiments = [], [], [], []

    print(f"Processing {len(subset)} tickets via OpenRouter...")

    for index, row in subset.iterrows():
        product = clean_text(
            row.get("Product Purchased", row.get("ProductPurchased")), "Unknown Product"
        )
        subject = clean_text(row.get("Ticket Subject"), "General Inquiry")
        description = clean_text(row.get("Ticket Description"), "")

        prompt = build_prompt(product, subject, description)

        try:
            triage = triage_ticket(prompt)

            urgencies.append(triage.ai_urgency)
            categories.append(triage.ai_category)
            escalations.append(triage.requires_immediate_escalation)  # real bool
            sentiments.append(triage.summary_sentiment)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            urgencies.append("Error")
            categories.append("Error")
            escalations.append(False)
            sentiments.append("Error")

    subset["AI_Urgency"] = urgencies
    subset["AI_Category"] = categories
    subset["AI_Escalate"] = escalations
    subset["AI_Sentiment"] = sentiments

    subset.to_csv(output_path, index=False)
    print(f"\nAI enrichment complete! Saved output to {output_path}")
    print(subset[["Ticket ID", "AI_Urgency", "AI_Category", "AI_Escalate", "AI_Sentiment"]])

if __name__ == "__main__":
    enrich_support_tickets()
