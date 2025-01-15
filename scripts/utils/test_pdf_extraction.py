import sys
import os

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from scripts.utils.pdf_processing import extract_text_from_folder



# Define input and output folders
input_folder = "Test/input"
output_folder = "Test/output"

# Extract text from PDFs
extract_text_from_folder(input_folder, output_folder)