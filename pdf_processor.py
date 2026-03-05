import os
import fitz
from config import PROJECT_DATA_ROOT, OUTPUT_TXT_ROOT, PDF_EXTENSION, TEXT_EXTENSION
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_pdf_to_text(pdf_path, output_txt_path):
    """
    Converts a single PDF file to a text file using PyMuPDF.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_txt_path (str): Path where the output text file will be saved.
    """
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    fitz.TOOLS.reset_mupdf_warnings()

    try:
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        text = ""

        with fitz.open(pdf_path) as doc:
            if doc.is_dirty:
                logger.warning(f"PDF was repaired upon opening: {pdf_path}")

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
        
        stderr_output = fitz.TOOLS.mupdf_warnings()
        if stderr_output:
            logger.warning(f"MuPDF error/warning for {pdf_path}: {stderr_output.strip()}")

        if text == "":
            logger.warning(f"No text extracted for {pdf_path}")
        
                
        with open(output_txt_path, "w", encoding='utf-8') as txt_file:
            txt_file.write(text)
        
        logger.info(f"Converted {pdf_path} into {output_txt_path} with length of text: {len(text)}")
        doc.close()
        return True, pdf_path
    
    except Exception as e:
        logger.error(f"PyMuPDF failed for converting {pdf_path}: {e}")
        return False, pdf_path

    finally:
        fitz.TOOLS.mupdf_display_errors(True)
        fitz.TOOLS.mupdf_display_warnings(True)
        fitz.TOOLS.reset_mupdf_warnings()
  
def _prepare_task(filename, dirpath, input_root_dir, output_root_dir):
    """
        Helper function
    """
    if filename.lower().endswith(PDF_EXTENSION):
        pdf_file_path = os.path.join(dirpath, filename)
        relative_path = os.path.relpath(dirpath, input_root_dir)
        output_dir = os.path.join(output_root_dir, relative_path)
        txt_filename = os.path.splitext(filename)[0] + TEXT_EXTENSION
        output_txt_path = os.path.join(output_dir, txt_filename)
        return (pdf_file_path, output_txt_path)
    return None
    

def process_all_pdfs(input_root_dir=PROJECT_DATA_ROOT, output_root_dir=OUTPUT_TXT_ROOT, max_workers=None):
    """
    Recursively walks through the input directory, finds PDFs,
    and converts them to text files, preserving the folder structure.

    Args:
        input_root_dir (str): Root directory containing PDF files (e.g., 'Dương', 'Giang').
        output_root_dir (str): Root directory where text files will be saved.
    """
    logger.info(f"Starting PDF processing from '{input_root_dir}'")
    logger.info(f"Text files will be saved to '{output_root_dir}'")
    tasks = []

    for dirpath, dirnames, filenames in os.walk(input_root_dir):
        for filename in filenames:
                task_args = _prepare_task(filename, dirpath, input_root_dir, output_root_dir)
                if task_args:
                    tasks.append(task_args)
    
    logger.info(f"Found {len(tasks)} PDF files to convert. Multiprocessing starting...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(convert_pdf_to_text, pdf_path, txt_path): (pdf_path, txt_path)
            for pdf_path, txt_path in tasks
        }

        for future in as_completed(future_to_task):
            success, pdf_path = future.result()
            if not success:
                logger.error(f"Failed to convert {pdf_path}")

    logger.info("PDF processing completed.")

if __name__ == "__main__":
    process_all_pdfs()