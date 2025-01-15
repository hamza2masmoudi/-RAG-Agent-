import sys
import os
import faiss
import numpy as np
from scripts.utils.embedding_generation import generate_embeddings

# Automatically add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize FAISS index and storage
dimension = 384  # Embedding dimension size
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
embedding_to_text = {}  # Dictionary to map FAISS indices to text

# Clear FAISS index before running
index.reset()

def add_real_embeddings_to_index(texts, index, text_mapping):
    """
    Generate embeddings for a list of texts and add them to the FAISS index.

    Args:
        texts (list): List of strings to embed and store.
        index (faiss.IndexFlatL2): The FAISS index object.
        text_mapping (dict): Dictionary mapping FAISS indices to original text.
    """
    for i, text in enumerate(texts):
        try:
            print(f"\nProcessing text {i + 1}/{len(texts)}: {text}")
            
            # Generate embeddings
            embedding = generate_embeddings(text)
            
            # Debugging: Check the embedding
            print(f"Generated embedding (first 10 values): {embedding[:10]}")
            print(f"Embedding shape: {len(embedding)}, dtype: {type(embedding[0])}")

            # Validate embedding shape
            if embedding is not None and len(embedding) == dimension:
                embedding_array = np.array([embedding], dtype='float32')
                print(f"Prepared embedding for FAISS (first 5 values): {embedding_array[:5]}...")
                
                # Add to FAISS index
                index.add(embedding_array)
                text_mapping[index.ntotal - 1] = text
                print(f"Successfully added to FAISS: {text}")
            else:
                print(f"Invalid embedding for text: {text}")
        except Exception as e:
            print(f"Error processing text {i + 1}: {e}")


# Define sample texts to add to FAISS
sample_texts = [
    "This is the first document.",
    "This is the second document.",
    "Here is another piece of text."
]

# Add sample texts to the FAISS index
print("Adding documents to FAISS index...")
add_real_embeddings_to_index(sample_texts, index, embedding_to_text)

# Test querying the FAISS index
query_text = "Find the first document."
print(f"\nQuerying FAISS index with: {query_text}")
try:
    # Generate query embedding
    query_embedding = generate_embeddings(query_text)
    print(f"Query Embedding shape: {len(query_embedding)}, dtype: {type(query_embedding[0])}")
    print(f"Query Embedding (first 10 values): {query_embedding[:10]}")
    
    # Validate query embedding
    if query_embedding is not None and len(query_embedding) == dimension:
        query_array = np.array([query_embedding], dtype='float32')
        print(f"Query Array (first 5 values): {query_array[:5]}...")
        
        # Perform FAISS search
        D, I = index.search(query_array, k=2)
        print("Distances:", D)
        print("Indices:", I)
        print("Query Results:", [embedding_to_text[i] for i in I[0]])
    else:
        print("Failed to generate embedding for query.")
except Exception as e:
    print(f"Error during query: {e}")