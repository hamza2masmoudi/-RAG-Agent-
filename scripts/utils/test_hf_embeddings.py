import sys
import os

# Automatically add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.utils.embedding_generation import generate_embeddings

# Example text
example_text = "This is a test sentence for generating embeddings."

# Generate embeddings
embeddings = generate_embeddings(example_text)

# Print the embedding size and a sample
print(f"Embedding size: {len(embeddings)}")
print(f"Embedding sample: {embeddings[:10]}")  # Print the first 10 values