import streamlit as st
import fitz  # PyMuPDF
import tiktoken
import numpy as np # For vector operations like dot product and norm
from openai import OpenAI # This is the new way to import the client

# --- Custom Embedding and Similarity Functions (Replacing embeddings_utils) ---

# Initialize the OpenAI client globally (or in a function if preferred, but only once)
# This client will automatically pick up the API key from st.secrets["OPENAI_API_KEY"]
# when deployed to Streamlit Cloud, or from environment variable OPENAI_API_KEY locally.
# If you explicitly want to use the sidebar input, you can pass it here:
# client = OpenAI(api_key=api_key) BUT this would re-initialize on every rerun, which is not ideal.
# Best practice is to use st.secrets.
# Let's initialize the client within a function for better control and to ensure API key is present.
@st.cache_resource # Cache the OpenAI client initialization
def get_openai_client(api_key_value):
    return OpenAI(api_key=api_key_value)

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generates an embedding for the given text using the OpenAI API.
    Handles the new OpenAI client (v1.x.x).
    """
    # Ensure API key is available before calling client
    if not st.session_state.get('openai_client'):
        st.error("OpenAI API client not initialized. Please provide an API key.")
        return None

    client = st.session_state.openai_client
    text = text.replace("\n", " ") # Recommended by OpenAI for consistent embeddings
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0 # Handle division by zero for zero vectors

    return dot_product / (norm_vec1 * norm_vec2)

# --- Streamlit App Layout and Logic ---

# Streamlit page config
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot with OpenAI")

# Sidebar for API key and file upload
st.sidebar.header("Configuration")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

# Initialize OpenAI client when API key is provided
if user_api_key:
    # Store the client in session state so it's persistent across reruns
    # and accessible by get_embedding.
    st.session_state.openai_client = get_openai_client(user_api_key)
    st.sidebar.success("OpenAI API Key set successfully!")
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop() # Stop execution until API key is provided

# Ensure fitz is installed (PyMuPDF)
try:
    import fitz
except ImportError:
    st.error("PyMuPDF (fitz) is not installed. Please add `PyMuPDF` to your requirements.txt.")
    st.stop()

# Ensure tiktoken is installed
try:
    import tiktoken
except ImportError:
    st.error("tiktoken is not installed. Please add `tiktoken` to your requirements.txt.")
    st.stop()


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
        # Ensure that `get_embedding` does not return None before encoding
        # This check is more for robustness if `get_embedding` had issues.
        # However, tiktoken does not depend on the OpenAI API directly for encoding.
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
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None # Will be set by get_openai_client

# Load and process PDF
if uploaded_file and st.session_state.openai_client:
    with st.spinner("Processing document..."):
        text = extract_text(uploaded_file)
        st.session_state.text_chunks = split_text(text)
    st.success("Document processed and ready for questions!")
elif not uploaded_file and st.session_state.text_chunks: # Allow using previously uploaded doc
    st.info("Document loaded from previous session. Upload a new one to change.")

# Chat interface
if st.session_state.text_chunks and st.session_state.openai_client:
    with st.chat_message("assistant"):
        st.markdown("Hi! Ask me anything about the uploaded document.")

    user_query = st.chat_input("Ask a question...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("Thinking..."):
            query_embedding = get_embedding(user_query, model="text-embedding-ada-002")

            if query_embedding is None:
                st.error("Could not generate embedding for your query. Please try again.")
                # Don't proceed if embedding failed
                st.session_state.chat_history.pop() # Remove user query if no embedding
            else:
                # Generate embeddings for chunks - this could be slow for large docs!
                # Consider caching chunk embeddings in session_state or a more robust solution.
                # For simplicity here, we re-embed if not cached.
                if 'chunk_embeddings' not in st.session_state or len(st.session_state.chunk_embeddings) != len(st.session_state.text_chunks):
                    st.session_state.chunk_embeddings = [
                        get_embedding(chunk, model="text-embedding-ada-002")
                        for chunk in st.session_state.text_chunks
                    ]
                    # Filter out any None embeddings if get_embedding failed for some chunks
                    st.session_state.chunk_embeddings = [emb for emb in st.session_state.chunk_embeddings if emb is not None]

                if not st.session_state.chunk_embeddings:
                    st.error("Failed to generate embeddings for document chunks. Cannot answer questions.")
                    st.session_state.chat_history.pop()
                else:
                    # Filter out corresponding text chunks if their embedding failed
                    valid_text_chunks = [st.session_state.text_chunks[i] for i, emb in enumerate(st.session_state.chunk_embeddings) if emb is not None]

                    similarities = [cosine_similarity(query_embedding, emb) for emb in st.session_state.chunk_embeddings]

                    if similarities: # Ensure there are similarities to find max from
                        best_chunk_index = similarities.index(max(similarities))
                        best_chunk = valid_text_chunks[best_chunk_index]

                        # Use the OpenAI client for chat completion
                        chat_client = st.session_state.openai_client
                        try:
                            response = chat_client.chat.completions.create( # Updated API call
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3
                            )
                            answer = response.choices[0].message.content # Access content directly from Pydantic model
                        except Exception as e:
                            st.error(f"Error generating AI response: {e}")
                            answer = "I'm sorry, I couldn't generate a response."
                    else:
                        answer = "I couldn't find a relevant section in the document to answer your question."

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat['role']):
            st.markdown(chat['content'])
else:
    if not st.session_state.openai_client:
        st.info("Please enter your OpenAI API key in the sidebar.")
    elif not uploaded_file:
        st.info("Please upload a PDF document and provide an API key to start.")