# main.py

import os
from config import PROJECT_DATA_ROOT, OUTPUT_TXT_ROOT, OUTPUT_TFIDF_ROOT, NEW_DATASET_FILE, NEW_DATASET_OUTPUT, NEW_FINAL_CSV_PATH
from pdf_processor import process_all_pdfs
from tfidf_calculator import calculate_tfidf
import logging
from causal_analysis import causal_main

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the Financial Report Analysis Pipeline")
    print("Please choose the task you want to execute: ")
    print("1. Converting PDFs to Text file")
    print("2. Calculate TF-IDF scores for the output_txt folder")
    print("3. Run Causal Machine Learning model and analysis")
    your_choice = int(input("Selection: "))

    # --- Step 1: PDF to Text Conversion ---
    if your_choice == 1:
        logger.info("--- Step 1: Converting PDFs to Text ---")
        if not os.path.exists(NEW_DATASET_FILE):
            logger.error(f"Input data directory '{NEW_DATASET_FILE}' does not exist. Please check the path in config.py.")
            return

        logger.info(f"Dataset input directory: {NEW_DATASET_FILE}")
        logger.info(f"Dataset output directory: {NEW_DATASET_OUTPUT}")
        process_all_pdfs(input_root_dir = NEW_DATASET_FILE, output_root_dir=NEW_DATASET_OUTPUT, max_workers=12)
        logger.info("--- PDF to Text Conversion Completed ---")


    elif your_choice == 2:
        logger.info("--- Step 2: Calculating TF-IDF Scores ---")    
        if not os.path.exists(NEW_DATASET_OUTPUT):
            logger.warning(f"Text output directory '{NEW_DATASET_OUTPUT}' does not exist. Did Step 1 run successfully?")
        
        logger.info(f"Dataset input: {NEW_DATASET_OUTPUT}")
        calculate_tfidf(input_txt_root=NEW_DATASET_OUTPUT)
        logger.info("--- TF-IDF Calculation Completed ---")    


    elif your_choice == 3:
        logger.info("--- Step 3: Causal Machine Learning Interference ---")
        # Please config csv path like this format in the config.py
        # final_csv_path = "/mnt/e/NEUConference/ClimateRisk/output_tfidf/Combined_Company_Data_2022_2024_Final2.csv"
        final_csv_path = NEW_FINAL_CSV_PATH
        if not os.path.exists(final_csv_path):
            logger.warning(f"Text output directory '{NEW_FINAL_CSV_PATH}' does not exist. Did Step 1 run successfully?")
        causal_main(final_csv_path)
        logger.info("--- Causal Machine Learning Interference Complete ---")    

if __name__ == "__main__":
    main()
