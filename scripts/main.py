# Main script for the RAG Agent
from scripts.utils.pdf_processing import extract_text_from_pdf
from scripts.utils.embedding_generation import generate_embeddings
from scripts.utils.query_handler import query_rag_system

if __name__ == "__main__":
    print("Welcome to the RAG Agent!")
    # Add your main logic here