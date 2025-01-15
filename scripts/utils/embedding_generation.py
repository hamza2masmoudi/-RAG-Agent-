from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from typing import List, Dict

# Load the pre-trained model and tokenizer (downloaded from Hugging Face)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Initialize FAISS index (L2 distance)
dimension = 384  # Dimension of embeddings (same as the model output)
index = faiss.IndexFlatL2(dimension)
embedding_to_text: Dict[int, str] = {}  # To map FAISS index to original text


def generate_embeddings(text: str) -> np.ndarray:
    """
    Generate embeddings for a given text using Hugging Face Transformers.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        np.ndarray: A NumPy array of embedding values in float32 format.
    """
    try:
        # Tokenize and process the text
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Perform mean pooling to get the sentence embedding
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

        # Ensure embeddings are on CPU and convert to float32 NumPy array
        embeddings_np = embeddings.cpu().numpy().astype('float32')
        return embeddings_np
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.zeros((384,), dtype='float32')  # Return a default embedding in case of error


def add_to_index(text: str, index, text_mapping: Dict[int, str]):
    """
    Generate embeddings for text and add them to the FAISS index.

    Args:
        text (str): The text to add to the index.
        index (faiss.IndexFlatL2): The FAISS index object.
        text_mapping (Dict[int, str]): Dictionary mapping index IDs to original text.
    """
    try:
        # Generate embedding
        embedding = generate_embeddings(text)
        print(f"Embedding for '{text}': {embedding[:10]}")  # Print first 10 values
        print(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")

        # Prepare embedding for FAISS
        embedding_array = np.array([embedding], dtype='float32')
        print(f"Prepared FAISS embedding: {embedding_array[:5]}")  # Print first 5 values

        # Add to FAISS index
        index.add(embedding_array)
        text_mapping[index.ntotal - 1] = text  # Map index ID to text
        print(f"Successfully added to FAISS: {text}")
    except Exception as e:
        print(f"Error adding to FAISS: {e}")

def query_index(query: str, index, text_mapping: Dict[int, str], top_k: int = 1) -> List[str]:
    """
    Query the FAISS index to find the most similar embeddings.

    Args:
        query (str): The query text.
        index (faiss.IndexFlatL2): The FAISS index object.
        text_mapping (Dict[int, str]): Dictionary mapping index IDs to original text.
        top_k (int): Number of top results to return.

    Returns:
        List[str]: A list of the most similar texts.
    """
    try:
        # Generate query embedding
        query_embedding = generate_embeddings(query)
        print(f"Query embedding: {query_embedding[:10]}")
        print(f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")

        # Prepare query for FAISS
        query_array = np.array([query_embedding], dtype='float32')
        print(f"Prepared query array: {query_array[:5]}")

        # Perform FAISS search
        distances, indices = index.search(query_array, top_k)
        print(f"Search distances: {distances}")
        print(f"Search indices: {indices}")

        # Retrieve results
        results = [text_mapping[i] for i in indices[0] if i != -1]
        return results
    except Exception as e:
        print(f"Error querying FAISS: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Add some sample texts to the index
    add_to_index("This is the first document.", index, embedding_to_text)
    add_to_index("This is the second document.", index, embedding_to_text)
    add_to_index("Here is another piece of text.", index, embedding_to_text)

    # Query the index
    query_text = "Find the first document."
    results = query_index(query_text, index, embedding_to_text, top_k=2)
    print("Query Results:", results)