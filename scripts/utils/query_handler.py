# Utility functions for querying the RAG system
import numpy as np

def query_rag_system(user_query, index, documents):
    """Handles querying the RAG system."""
    # Generate embedding for the query
    query_embedding = generate_embeddings(user_query)
    
    # Search for the most similar document in the index
    D, I = index.search(np.array([query_embedding], dtype='float32'), k=1)
    
    # Retrieve the corresponding document
    relevant_doc = documents[I[0][0]]
    
    return relevant_doc