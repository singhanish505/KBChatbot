import streamlit as st
import fitz  # PyMuPDF
import openai
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity

# Streamlit page config
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot with OpenAI")

# Sidebar for API key and file upload
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

# Set API Key
if api_key:
    openai.api_key = api_key
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")

# Extract and chunk text from PDF
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, max_tokens=500):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    words = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in words:
        sentence = sentence.strip()
        if len(enc.encode(current_chunk + sentence)) < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    chunks.append(current_chunk.strip())
    return chunks

# Store chat history and state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []

# Load and process PDF
if uploaded_file and api_key:
    with st.spinner("Processing document..."):
        text = extract_text(uploaded_file)
        st.session_state.text_chunks = split_text(text)
    st.success("Document processed and ready for questions!")

# Chat interface
if st.session_state.text_chunks:
    with st.chat_message("assistant"):
        st.markdown("Hi! Ask me anything about the uploaded document.")

    user_query = st.chat_input("Ask a question...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("Thinking..."):
            query_embedding = get_embedding(user_query, engine="text-embedding-ada-002")
            chunk_embeddings = [get_embedding(chunk, engine="text-embedding-ada-002") for chunk in st.session_state.text_chunks]
            similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
            best_chunk = st.session_state.text_chunks[similarities.index(max(similarities))]

            prompt = f"Answer the following question based on the context:\n\nContext:\n{best_chunk}\n\nQuestion:\n{user_query}\n\nAnswer:"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message['content']

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])
else:
    st.info("Please upload a PDF document and provide an API key to start.")
