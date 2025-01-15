import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def extract_text_from_folder(folder_path, output_folder):
    """
    Extracts text from all PDF files in a folder and saves the text files.

    Args:
        folder_path (str): Path to the folder containing PDF files.
        output_folder (str): Path to save the extracted text files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Extracted text saved to {output_file}")
            else:
                print(f"No text extracted from {filename}")