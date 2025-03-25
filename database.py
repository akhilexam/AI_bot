from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

# Initialize Supabase Client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_all_documents():
    """Retrieve all Q&A documents from Supabase."""
    response = supabase.table("documents").select("question, answer, embedding").execute()
    return response.data if response.data else []

def add_document(question, answer, embedding):
    """Insert a new question-answer pair into Supabase."""
    data = {"question": question, "answer": answer, "embedding": embedding}
    supabase.table("documents").insert(data).execute()
