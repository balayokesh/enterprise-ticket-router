import pandas as pd
import os

def generate_ticket_report(input_path="enriched_data.csv", output_summary_path="ticket_summary_report.csv"):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the AI enrichment module first.")
        return
        
    print("Loading enriched dataset...")
    df = pd.read_csv(input_path)
    
    total_tickets = len(df)
    print(f"\n--- SUPPORT TICKET PIPELINE METRICS (Total Tickets: {total_tickets}) ---")
    
    # 1. Escalation Summary
    if 'AI_Escalate' in df.columns:
        escalations = df['AI_Escalate'].sum()
        print(f"\n🚨 Immediate Escalations Required: {escalations} out of {total_tickets}")
    
    # 2. Urgency Breakdown
    if 'AI_Urgency' in df.columns:
        print("\n📊 Breakdown by Urgency:")
        urgency_counts = df['AI_Urgency'].value_counts()
        print(urgency_counts.to_string())
        
    # 3. Category Breakdown
    if 'AI_Category' in df.columns:
        print("\n🏷️ Breakdown by Category:")
        category_counts = df['AI_Category'].value_counts()
        print(category_counts.to_string())
        
    # 4. Sentiment Distribution
    if 'AI_Sentiment' in df.columns:
        print("\n💬 Customer Sentiment Analysis:")
        sentiment_counts = df['AI_Sentiment'].value_counts()
        print(sentiment_counts.to_string())
        
    # Exporting a consolidated summary CSV
    summary_data = {
        "Metric": ["Total Tickets", "Total Escalations"],
        "Value": [total_tickets, escalations if 'AI_Escalate' in df.columns else 0]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_summary_path, index=False)
    print(f"\nSaved executive summary report to {output_summary_path}")

if __name__ == "__main__":
    generate_ticket_report()
