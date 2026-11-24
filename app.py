import os
import tempfile
import traceback
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from ai_enrichment import enrich_support_tickets
from pipeline import clean_support_data

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-triage-dashboard-secret")

BASE_DIR = Path(__file__).resolve().parent
ENRICHED_CSV = BASE_DIR / "enriched_data.csv"
TELEMETRY_CSV = BASE_DIR / "telemetry_metrics.csv"


def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df.fillna("")
    except Exception:
        return pd.DataFrame()


@app.route("/", methods=["GET"])
def index():
    urgency_filter = request.args.get("urgency", "").strip()
    category_filter = request.args.get("category", "").strip()
    sentiment_filter = request.args.get("sentiment", "").strip()
    escalate_only = request.args.get("escalate_only", "") == "1"
    search_query = request.args.get("q", "").strip().lower()

    enriched_df = load_csv_safe(ENRICHED_CSV)
    telemetry_df = load_csv_safe(TELEMETRY_CSV)

    # Distinct filter options from full enriched dataset
    urgency_options = sorted([v for v in enriched_df["AI_Urgency"].unique() if v]) if "AI_Urgency" in enriched_df.columns else []
    category_options = sorted([v for v in enriched_df["AI_Category"].unique() if v]) if "AI_Category" in enriched_df.columns else []
    sentiment_options = sorted([v for v in enriched_df["AI_Sentiment"].unique() if v]) if "AI_Sentiment" in enriched_df.columns else []

    # Apply filters for Triage Results view
    filtered_df = enriched_df.copy()
    if not filtered_df.empty:
        if urgency_filter and "AI_Urgency" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["AI_Urgency"] == urgency_filter]
        if category_filter and "AI_Category" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["AI_Category"] == category_filter]
        if sentiment_filter and "AI_Sentiment" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["AI_Sentiment"] == sentiment_filter]
        if escalate_only and "AI_Escalate" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["AI_Escalate"].astype(str).str.lower() == "true"]
        if search_query:
            text_cols = [c for c in ["Ticket ID", "Ticket Subject", "Ticket Description", "Product Purchased", "ProductPurchased"] if c in filtered_df.columns]
            if text_cols:
                mask = pd.Series(False, index=filtered_df.index)
                for col in text_cols:
                    mask = mask | filtered_df[col].astype(str).str.lower().str.contains(search_query, na=False)
                filtered_df = filtered_df[mask]

    # Business KPIs
    total_tickets = len(enriched_df)
    high_urgency_count = (
        int((enriched_df["AI_Urgency"] == "High").sum())
        if not enriched_df.empty and "AI_Urgency" in enriched_df.columns
        else 0
    )
    escalation_count = (
        int((enriched_df["AI_Escalate"].astype(str).str.lower() == "true").sum())
        if not enriched_df.empty and "AI_Escalate" in enriched_df.columns
        else 0
    )

    # Telemetry KPIs
    total_tokens = 0
    total_cost = 0.0
    avg_latency = 0.0
    if not telemetry_df.empty:
        if "Total_Tokens" in telemetry_df.columns:
            total_tokens = int(pd.to_numeric(telemetry_df["Total_Tokens"], errors="coerce").fillna(0).sum())
        if "Estimated_Cost_USD" in telemetry_df.columns:
            total_cost = float(pd.to_numeric(telemetry_df["Estimated_Cost_USD"], errors="coerce").fillna(0).sum())
        if "Latency_Sec" in telemetry_df.columns:
            avg_latency = float(pd.to_numeric(telemetry_df["Latency_Sec"], errors="coerce").fillna(0).mean())

    triage_rows = filtered_df.to_dict(orient="records") if not filtered_df.empty else []
    triage_columns = list(filtered_df.columns) if not filtered_df.empty else []

    telemetry_rows = telemetry_df.to_dict(orient="records") if not telemetry_df.empty else []
    telemetry_columns = list(telemetry_df.columns) if not telemetry_df.empty else []

    return render_template(
        "index.html",
        total_tickets=total_tickets,
        filtered_count=len(triage_rows),
        high_urgency_count=high_urgency_count,
        escalation_count=escalation_count,
        total_tokens=total_tokens,
        total_cost=total_cost,
        avg_latency=avg_latency,
        urgency_options=urgency_options,
        category_options=category_options,
        sentiment_options=sentiment_options,
        selected_urgency=urgency_filter,
        selected_category=category_filter,
        selected_sentiment=sentiment_filter,
        escalate_only=escalate_only,
        search_query=search_query,
        triage_rows=triage_rows,
        triage_columns=triage_columns,
        telemetry_rows=telemetry_rows,
        telemetry_columns=telemetry_columns,
    )


def _is_csv_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() == ".csv"


@app.route("/run", methods=["POST"])
def run_pipeline():
    uploaded_file = request.files.get("csv_file")
    if not uploaded_file or not uploaded_file.filename:
        flash("Please select a .csv file to upload.", "error")
        return redirect(url_for("index"))

    original_name = uploaded_file.filename.strip()
    if not _is_csv_filename(original_name):
        flash("Invalid file type. Only .csv files are allowed.", "error")
        return redirect(url_for("index"))

    safe_name = secure_filename(original_name) or "input.csv"
    if not safe_name.lower().endswith(".csv"):
        safe_name = f"{safe_name}.csv"

    try:
        raw_limit = request.form.get("limit", "5")
        limit = max(1, min(int(raw_limit), 25))  # Cap at 25 from UI for safe testing
    except ValueError:
        limit = 5

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_csv_path = Path(tmp_dir) / safe_name
            cleaned_csv_path = Path(tmp_dir) / f"cleaned_{safe_name}"
            uploaded_file.save(raw_csv_path)

            preview_df = pd.read_csv(raw_csv_path, nrows=1)
            if preview_df.empty:
                flash("Uploaded CSV file contains no rows.", "error")
                return redirect(url_for("index"))

            clean_support_data(
                input_path=str(raw_csv_path),
                output_path=str(cleaned_csv_path),
            )

            enrich_support_tickets(
                input_path=str(cleaned_csv_path),
                output_path=str(ENRICHED_CSV),
                telemetry_path=str(TELEMETRY_CSV),
                limit=limit,
            )
        flash(f"Pipeline completed for {limit} ticket(s) from {original_name}. Triage and telemetry views updated.", "success")
    except Exception as exc:
        traceback.print_exc()
        flash(f"Pipeline run failed: {exc}", "error")

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
