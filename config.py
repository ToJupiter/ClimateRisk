import os

# --- Directory Paths ---
# Assuming the script runs from the project root
# Adjust these paths according to your actual setup

# Base directory where the zipped/cloned project resides
# This is the parent directory of folders like 'Dương', 'Giang', etc.
PROJECT_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Adjust if needed

# Output directory for the converted .txt files
OUTPUT_TXT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output_txt'))

# --- File Extensions ---
PDF_EXTENSION = '.pdf'
TEXT_EXTENSION = '.txt'