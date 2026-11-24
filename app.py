import os
import tempfile
import threading
import traceback
from pathlib import Path

import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from ai_enrichment import enrich_support_tickets
from pipeline import clean_support_data

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-triage-dashboard-secret")

BASE_DIR = Path(__file__).resolve().parent
ENRICHED_CSV = BASE_DIR / "enriched_data.csv"
TELEMETRY_CSV = BASE_DIR / "telemetry_metrics.csv"
CSV_WRITE_LOCK = threading.Lock()

REVIEW_STATUSES = ("Pending Review", "Approved & Sent", "Edited & Sent")
HIDDEN_TABLE_COLUMNS = {"Suggested_Reply", "Cited_Sources"}


def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df.fillna("")
    except Exception:
        return pd.DataFrame()


def normalize_enriched_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "Suggested_Reply" not in df.columns:
        df["Suggested_Reply"] = ""
    else:
        df["Suggested_Reply"] = df["Suggested_Reply"].fillna("").astype(str)

    if "Cited_Sources" not in df.columns:
        df["Cited_Sources"] = "None"
    else:
        cited = df["Cited_Sources"].fillna("").astype(str).str.strip()
        df["Cited_Sources"] = cited.where(cited != "", "None")

    if "Review_Status" not in df.columns:
        df["Review_Status"] = "Pending Review"
    else:
        status = df["Review_Status"].fillna("").astype(str).str.strip()
        df["Review_Status"] = status.where(status != "", "Pending Review")
    return df


def compute_review_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = {status: 0 for status in REVIEW_STATUSES}
    if not df.empty and "Review_Status" in df.columns:
        vc = df["Review_Status"].astype(str).str.strip().value_counts()
        for status in REVIEW_STATUSES:
            counts[status] = int(vc.get(status, 0))
    return counts


def _matches_ticket_id(series: pd.Series, ticket_id: str) -> pd.Series:
    target = str(ticket_id).strip()
    s = series.astype(str).str.strip()
    mask = s == target
    if not mask.any():
        try:
            target_num = int(target)
            numeric_vals = pd.to_numeric(s, errors="coerce")
            mask = numeric_vals == target_num
        except ValueError:
            pass
    return mask


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


@app.route("/", methods=["GET"])
def index():
    urgency_filter = request.args.get("urgency", "").strip()
    category_filter = request.args.get("category", "").strip()
    sentiment_filter = request.args.get("sentiment", "").strip()
    review_status_filter = request.args.get("review_status", "").strip()
    escalate_only = request.args.get("escalate_only", "") == "1"
    search_query = request.args.get("q", "").strip().lower()

    enriched_df = normalize_enriched_df(load_csv_safe(ENRICHED_CSV))
    telemetry_df = load_csv_safe(TELEMETRY_CSV)

    # Distinct filter options from full enriched dataset
    urgency_options = sorted([v for v in enriched_df["AI_Urgency"].unique() if v]) if "AI_Urgency" in enriched_df.columns else []
    category_options = sorted([v for v in enriched_df["AI_Category"].unique() if v]) if "AI_Category" in enriched_df.columns else []
    sentiment_options = sorted([v for v in enriched_df["AI_Sentiment"].unique() if v]) if "AI_Sentiment" in enriched_df.columns else []
    existing_review_statuses = [v for v in enriched_df["Review_Status"].unique() if v] if "Review_Status" in enriched_df.columns else []
    review_status_options = list(REVIEW_STATUSES) + sorted([v for v in existing_review_statuses if v not in REVIEW_STATUSES])

    # Apply filters for Triage Results view
    filtered_df = enriched_df.copy()
    if not filtered_df.empty:
        if urgency_filter and "AI_Urgency" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["AI_Urgency"] == urgency_filter]
        if category_filter and "AI_Category" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["AI_Category"] == category_filter]
        if sentiment_filter and "AI_Sentiment" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["AI_Sentiment"] == sentiment_filter]
        if review_status_filter and "Review_Status" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Review_Status"] == review_status_filter]
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
    review_counts = compute_review_counts(enriched_df)

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
    triage_columns = [c for c in filtered_df.columns if c not in HIDDEN_TABLE_COLUMNS] if not filtered_df.empty else []

    telemetry_rows = telemetry_df.to_dict(orient="records") if not telemetry_df.empty else []
    telemetry_columns = list(telemetry_df.columns) if not telemetry_df.empty else []

    return render_template(
        "index.html",
        total_tickets=total_tickets,
        filtered_count=len(triage_rows),
        high_urgency_count=high_urgency_count,
        escalation_count=escalation_count,
        review_counts=review_counts,
        total_tokens=total_tokens,
        total_cost=total_cost,
        avg_latency=avg_latency,
        urgency_options=urgency_options,
        category_options=category_options,
        sentiment_options=sentiment_options,
        review_status_options=review_status_options,
        selected_urgency=urgency_filter,
        selected_category=category_filter,
        selected_sentiment=sentiment_filter,
        selected_review_status=review_status_filter,
        escalate_only=escalate_only,
        search_query=search_query,
        triage_rows=triage_rows,
        triage_columns=triage_columns,
        telemetry_rows=telemetry_rows,
        telemetry_columns=telemetry_columns,
    )


@app.route("/tickets/<ticket_id>/review", methods=["POST"])
def review_ticket(ticket_id: str):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    action = str(payload.get("action", "")).strip().lower()
    if action not in {"approve", "edit"}:
        return jsonify({"error": "Invalid action. Expected 'approve' or 'edit'."}), 400

    edited_reply = None
    if action == "edit":
        raw_reply = payload.get("suggested_reply")
        if not isinstance(raw_reply, str) or not raw_reply.strip():
            return jsonify({"error": "suggested_reply must be a non-empty string when action is 'edit'."}), 400
        edited_reply = raw_reply.strip()

    with CSV_WRITE_LOCK:
        if not ENRICHED_CSV.exists():
            return jsonify({"error": "Enriched dataset not found."}), 404

        try:
            df = pd.read_csv(ENRICHED_CSV, dtype=str, keep_default_na=False)
        except Exception as exc:
            return jsonify({"error": f"Failed to read enriched dataset: {exc}"}), 500

        if df.empty or "Ticket ID" not in df.columns:
            return jsonify({"error": f"Ticket '{ticket_id}' not found."}), 404

        df = normalize_enriched_df(df)
        mask = _matches_ticket_id(df["Ticket ID"], ticket_id)
        if not mask.any():
            return jsonify({"error": f"Ticket '{ticket_id}' not found."}), 404

        if action == "approve":
            df.loc[mask, "Review_Status"] = "Approved & Sent"
        else:
            df.loc[mask, "Suggested_Reply"] = edited_reply
            df.loc[mask, "Review_Status"] = "Edited & Sent"

        try:
            _atomic_write_csv(df, ENRICHED_CSV)
        except Exception as exc:
            return jsonify({"error": f"Failed to persist review update: {exc}"}), 500

        updated_ticket = df.loc[mask].iloc[0].to_dict()
        review_counts = compute_review_counts(df)

    return jsonify(
        {
            "status": "ok",
            "ticket": updated_ticket,
            "review_counts": review_counts,
        }
    ), 200


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
