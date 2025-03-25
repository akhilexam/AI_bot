import streamlit as st
from database import add_document, fetch_all_documents  # Ensure these functions are correctly implemented
from embedding import get_embedding, find_best_match  # Ensure these are correct

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
        try:
            embedding = get_embedding(new_question).tolist()  # Convert to list for JSON storage
            add_document(new_question, new_answer, embedding)  # Add to Supabase
            st.success("Data added successfully!")
        except Exception as e:
            st.error(f"Error adding data: {e}")
    else:
        st.warning("Please provide both question and answer.")

