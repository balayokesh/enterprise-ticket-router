import sys
import logging
from pipeline import clean_support_data  # Adjust function name if named differently in your ingestion/pipeline module
from ai_enrichment import enrich_support_tickets
from reporting import generate_ticket_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    logging.info("Starting Automated Support Ticket Triage & Analytics Pipeline...")
    
    try:
        # Step 1: Data Ingestion & Cleaning
        logging.info("--- Step 1/3: Running Data Ingestion & Cleaning ---")
        clean_support_data()
        
        # Step 2: AI Enrichment & Triage
        logging.info("--- Step 2/3: Running AI Enrichment & Triage ---")
        enrich_support_tickets()
        
        # Step 3: Analytics & Reporting
        logging.info("--- Step 3/3: Generating Executive Reports ---")
        generate_ticket_report()
        
        logging.info("Pipeline execution completed successfully! 🎉")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()