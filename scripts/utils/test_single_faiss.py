import faiss
import numpy as np
from scripts.utils.embedding_generation import generate_embeddings

# Initialize FAISS index
dimension = 384
index = faiss.IndexFlatL2(dimension)
embedding_to_text = {}

# Add embedding to FAISS
try:
    text = "This is the first document."
    embedding = generate_embeddings(text)

    # Debugging: Print embedding details
    print(f"Generated Embedding for '{text}': {embedding[:10]}")  # First 10 values
    print(f"Shape: {embedding.shape}, Dtype: {embedding.dtype}")

    # Prepare and add to FAISS
    embedding_array = np.array([embedding], dtype='float32')  # Convert to 2D array
    print(f"Prepared FAISS Embedding: {embedding_array[:5]}")  # First 5 values
    index.add(embedding_array)
    embedding_to_text[index.ntotal - 1] = text
    print(f"Successfully added to FAISS: {text}")
except Exception as e:
    print(f"Error adding embedding to FAISS: {e}")

# Query FAISS index
try:
    query_text = "This is the first document."
    query_embedding = generate_embeddings(query_text)

    # Debugging: Print query embedding details
    print(f"Query Embedding: {query_embedding[:10]}")  # First 10 values
    print(f"Shape: {query_embedding.shape}, Dtype: {query_embedding.dtype}")

    # Prepare and query FAISS
    query_array = np.array([query_embedding], dtype='float32')  # Convert to 2D array
    print(f"Prepared Query Array: {query_array[:5]}")  # First 5 values
    D, I = index.search(query_array, k=1)  # Perform the search
    print("Distances:", D)
    print("Indices:", I)

    # Retrieve and print query results
    results = [embedding_to_text.get(i, "Index not found") for i in I[0]]
    print("Query Results:", results)
except Exception as e:
    print(f"Error querying FAISS: {e}")