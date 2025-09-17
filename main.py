# main.py

import os
from config import PROJECT_DATA_ROOT, OUTPUT_TXT_ROOT, OUTPUT_TFIDF_ROOT, NEW_DATASET_FILE, NEW_DATASET_OUTPUT
from pdf_processor import process_all_pdfs
from tfidf_calculator import calculate_tfidf
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the entire pipeline.
    Currently, it only handles PDF to text conversion.
    """
    logger.info("Starting the Financial Report Analysis Pipeline")

    # --- Step 1: PDF to Text Conversion ---
    logger.info("--- Step 1: Converting PDFs to Text ---")
    # if not os.path.exists(PROJECT_DATA_ROOT):
    #     logger.error(f"Input data directory '{PROJECT_DATA_ROOT}' does not exist. Please check the path in config.py.")
    #     return

    #Turn on if pdfs are not processed
    # process_all_pdfs(input_root_dir = NEW_DATASET_FILE, output_root_dir=NEW_DATASET_OUTPUT, max_workers=12)
    logger.info("--- PDF to Text Conversion Completed ---")

    logger.info("--- Step 2: Calculating TF-IDF Scores ---")
    # if not os.path.exists(OUTPUT_TXT_ROOT):
    #     logger.warning(f"Text output directory '{OUTPUT_TXT_ROOT}' does not exist. Did Step 1 run successfully?")
    calculate_tfidf(input_txt_root=NEW_DATASET_OUTPUT)
    
    # calculate_tfidf()
    logger.info("--- TF-IDF Calculation Completed ---")    
    # logger.info("Pipeline execution finished (for now, only PDF conversion).")

if __name__ == "__main__":
    main()
