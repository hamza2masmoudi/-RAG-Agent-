from scripts.utils.embedding_generation import generate_embeddings
import numpy as np

# Test embedding generation for a simple text
text = "This is the first document."
embedding = generate_embeddings(text)

# Validate the output
print(f"Generated Embedding: {embedding[:10]}")  # First 10 values
print(f"Shape: {embedding.shape}, Dtype: {embedding.dtype}")

# Ensure it matches FAISS requirements
assert isinstance(embedding, np.ndarray), "Embedding is not a NumPy array."
assert embedding.dtype == np.float32, "Embedding is not float32."
assert embedding.shape == (384,), "Embedding shape is not (384,)."