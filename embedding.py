import openai
import numpy as np
from config import OPENAI_API_KEY
from database import fetch_all_documents  # Ensure this function is properly defined

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    """
    Generate an embedding vector using OpenAI.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)  # Extract the embedding vector

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
