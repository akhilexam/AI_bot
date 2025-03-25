import streamlit as st
from app.database import add_document, fetch_all_documents
from app.embeddings import get_embedding, find_best_match

st.title("AI-Powered Q&A Bot with Supabase 🚀")
st.write("Ask any question, and I'll find the best answer!")

# ✅ User Q&A Interface
user_input = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if user_input:
        answer = find_best_match(user_input)
        st.write("**Answer:**", answer)
    else:
        st.warning("Please enter a question.")

# ✅ Admin Panel for Adding Data
st.subheader("Admin Panel: Add Data")
new_question = st.text_input("New Question:")
new_answer = st.text_area("Answer:")
if st.button("Add to Database"):
    if new_question and new_answer:
        embedding = get_embedding(new_question).tolist()
        add_document(new_question, new_answer, embedding)
        st.success("Data added successfully!")
    else:
        st.warning("Please provide both question and answer.")
