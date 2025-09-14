# tfidf_calculator.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
import logging
import os
from config import OUTPUT_TXT_ROOT, OUTPUT_TFIDF_ROOT, CLIMATE_RISK_KEYWORDS, TFIDF_CSV_FILE
import ssl
import concurrent.futures
from functools import partial

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


## NLTK

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#### Stop_words processing
try:
    STOP_WORDS = set(stopwords.words('english'))

except LookupError:
    logger.warning("NLTK stopwords not found. Downloading 'stopwords' corpus...")
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))

os.makedirs(OUTPUT_TFIDF_ROOT, exist_ok=True)

def preprocess_text(text):
    """
    Preprocesses the text for TF-IDF calculation.
    - Lowercases the text.
    - Removes punctuation and numbers.
    - Tokenizes (implicitly handled by TfidfVectorizer).
    - Removes stop words.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def process_single_file(args):
    """
    Processes a single text file: reads, preprocesses, and generates doc_id.
    Args:
        args (tuple): (txt_file_path, input_txt_root)
    Returns:
        tuple: (processed_text, doc_id) or (None, None) on error
    """
    txt_file_path, input_txt_root = args
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        processed_text = preprocess_text(text)

        relative_dir = os.path.relpath(os.path.dirname(txt_file_path), input_txt_root)
        doc_name = os.path.splitext(os.path.basename(txt_file_path))[0]
        doc_id = os.path.join(relative_dir, doc_name).replace(os.sep, '/')

        return processed_text, doc_id
    except Exception as e:
        logger.error(f"Error processing file '{txt_file_path}': {e}")
        return None, None

def calculate_tfidf(input_txt_root=OUTPUT_TXT_ROOT, output_tfidf_root=OUTPUT_TFIDF_ROOT):
    """
    Calculates TF-IDF scores for predefined keywords across all text files.

    Args:
        input_txt_root (str): Root directory containing the .txt files.
        output_tfidf_root (str): Directory to save the TF-IDF results.
    """
    logger.info(f"Starting TF-IDF calculation on text files from '{input_txt_root}'")

    file_paths = []
    logger.info("Collecting text file paths...")    
    for dirpath, _, filenames in os.walk(input_txt_root):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                txt_file_path = os.path.join(dirpath, filename)
                file_paths.append((txt_file_path, input_txt_root))
    
    if not file_paths:
        logger.error("No text documents for TF-IDF")
        return
    
    documents = []
    doc_ids = []
    worker_func = partial(process_single_file)

    logger.info("Starting multiprocessing for text loading and preprocessing...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        future_to_path = {executor.submit(worker_func, path_tuple): path_tuple for path_tuple in file_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path_tuple = future_to_path[future]
            try:
                processed_text, doc_id = future.result()
                if processed_text is not None and doc_id is not None:
                    documents.append(processed_text)
                    doc_ids.append(doc_id)
            
            except Exception as exc:
                logger.error(f'File {path_tuple[0]} generated an exception: {exc}')

    logger.info(f"Finished multiprocessing. Successfully processed {len(documents)} documents.")

    if not documents:
        logger.error("No documents were successfully processed.")
        return


    vectorizer = TfidfVectorizer(
        vocabulary=CLIMATE_RISK_KEYWORDS,
        stop_words='english',
        ngram_range=(1,2)
    )
    
    logger.info("Fitting TfidfVectorizer and transforming documents...")
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as e:
        logger.error(f"Error during TF-IDF calculation: {e}. Check if vocabulary keywords are present in the corpus.")
        return

    logger.info("Creating TF-IDF dataframe...")
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=doc_ids)
    df_tfidf.reset_index(inplace=True)
    df_tfidf.rename(columns={'index': 'doc_id'}, inplace=True)

    logger.info("Calculating 'ClimateRiskScore' (sum of TF-IDF scores)...")
    df_tfidf['ClimateRiskScore'] = df_tfidf[feature_names].sum(axis=1)

    output_file_path = os.path.join(output_tfidf_root, TFIDF_CSV_FILE)
    logger.info(f"Saving TF-IDF results to '{output_file_path}'...")
    df_tfidf.to_csv(output_file_path, index=False)
    logger.info("TF-IDF calculation and saving completed.")

if __name__ == "__main__":
    calculate_tfidf()