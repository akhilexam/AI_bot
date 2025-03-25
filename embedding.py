from sentence_transformers import SentenceTransformer
import numpy as np
from database import fetch_all_documents  # Ensure this function is properly defined

# Load the embedding model (runs locally)
model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient and accurate

def get_embedding(text):
    """
    Generate an embedding vector using SentenceTransformers.
    """
    return np.array(model.encode(text, normalize_embeddings=True))  # Normalize for better similarity

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_best_match(user_question):
    """
    Find the most relevant answer from Supabase based on similarity.
    """
    user_embedding = get_embedding(user_question)
    documents = fetch_all_documents()  # Fetch documents from Supabase

    best_match = None
    best_score = -1

    for doc in documents:
        doc_embedding = np.array(doc["embedding"])  # Convert stored embedding to NumPy array
        similarity = cosine_similarity(user_embedding, doc_embedding)

        if similarity > best_score:
            best_score = similarity
            best_match = doc

    return best_match["answer"] if best_match else "No relevant answer found."


