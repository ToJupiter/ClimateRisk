# main.py

import os
from config import PROJECT_DATA_ROOT, OUTPUT_TXT_ROOT
from pdf_processor import process_all_pdfs
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
    if not os.path.exists(PROJECT_DATA_ROOT):
        logger.error(f"Input data directory '{PROJECT_DATA_ROOT}' does not exist. Please check the path in config.py.")
        return

    #Turn on if pdfs are not processed
    # process_all_pdfs(max_workers=12)
    logger.info("--- PDF to Text Conversion Completed ---")

    # --- Future Steps (Placeholder) ---
    # logger.info("--- Step 2: Calculating TF-IDF ---")
    # calculate_tfidf(...) # Call function from tfidf_calculator.py
    #
    # logger.info("--- Step 3: Merging Data ---")
    # merge_data(...) # Call function from data_merger.py
    #
    # logger.info("--- Step 4: Training Models ---")
    # run_models(...) # Call function from model_trainer.py

    logger.info("Pipeline execution finished (for now, only PDF conversion).")

if __name__ == "__main__":
    main()
