import os
import fitz
from config import PROJECT_DATA_ROOT, OUTPUT_TXT_ROOT, PDF_EXTENSION, TEXT_EXTENSION
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import tempfile
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _extract_with_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text

def _extract_with_pdfplumber(pdf_path):
    import pdfplumber
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def _extract_with_pypdf2(pdf_path):
    import PyPDF2
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def _extract_with_pdftotext(pdf_path):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(['pdftotext', pdf_path, tmp_path], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            raise Exception(f"pdftotext failed: {result.stderr}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# def _extract_with_ocr(pdf_path):
#     import pytesseract
#     from pdf2image import convert_from_path
#     images = convert_from_path(pdf_path, dpi=200)
#     text = ""
#     for image in images:
#         text += pytesseract.image_to_string(image, lang='eng') + "\n"
#     return text

def convert_pdf_to_text(pdf_path, output_txt_path):
    try:
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        text = None
        methods = [
            ("pymupdf", _extract_with_pymupdf),
            ("pdfplumber", _extract_with_pdfplumber),
            ("pypdf2", _extract_with_pypdf2),
            ("pdftotext", _extract_with_pdftotext)
            # ("ocr", _extract_with_ocr)
        ]
        for method_name, method_func in methods:
            try:
                text = method_func(pdf_path)
                if text and text.strip():
                    # logger.info(f"Successfully extracted text from {pdf_path} using {method_name}")
                    break
            except Exception as e:
                # logger.warning(f"Method {method_name} failed for {pdf_path}: {e}")
                continue
        if not text or not text.strip():
            raise Exception("All extraction methods failed or returned empty text")
        with open(output_txt_path, "w", encoding='utf-8') as txt_file:
            txt_file.write(text)
        success_info = f"Converted {pdf_path} into {output_txt_path}"
        # logger.info(f"Converted {pdf_path} into {output_txt_path}")
        return True, pdf_path, success_info
    except Exception as e:
        error_info = f"Error converting {pdf_path}: {e}"
        # logger.error(f"Error converting {pdf_path}: {e}")
        return False, pdf_path, error_info
    
def _prepare_task(filename, dirpath, input_root_dir, output_root_dir):
    if filename.lower().endswith(PDF_EXTENSION):
        pdf_file_path = os.path.join(dirpath, filename)
        relative_path = os.path.relpath(dirpath, input_root_dir)
        output_dir = os.path.join(output_root_dir, relative_path)
        txt_filename = os.path.splitext(filename)[0] + TEXT_EXTENSION
        output_txt_path = os.path.join(output_dir, txt_filename)
        return (pdf_file_path, output_txt_path)
    return None
    

def process_all_pdfs(input_root_dir=PROJECT_DATA_ROOT, output_root_dir=OUTPUT_TXT_ROOT, max_workers=None):
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
            success, pdf_path, log_msg = future.result()
            if success:
                logger.info(log_msg)
            else:
                logger.error(log_msg)

    logger.info("PDF processing completed.")

if __name__ == "__main__":
    process_all_pdfs()