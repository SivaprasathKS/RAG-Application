import os
import streamlit as st
from google import genai
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

# ---------------------------
# Set up Google GenAI client
# ---------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------------------
# Set up Chroma client
# ---------------------------
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

collection_name = "documents"

existing_collections = [c.name for c in chroma_client.list_collections()]

if collection_name in existing_collections:
    collection = chroma_client.get_collection(name=collection_name)
else:
    collection = chroma_client.create_collection(name=collection_name)

# ---------------------------
# Helper Functions
# ---------------------------

def get_embedding(text: str):
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text]  # Correct for embedding
    )
    return response.embeddings[0].values


def chunk_text(text: str, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("📚 RAG Application using Google GenAI + ChromaDB")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# ---------------------------
# Upload & Process PDFs
# ---------------------------

if uploaded_files:

    # Completely reset collection (no duplicates)
    chroma_client.delete_collection(name=collection_name)
    collection = chroma_client.create_collection(name=collection_name)

    all_chunks = []
    all_ids = []
    all_metadata = []

    for file in uploaded_files:
        pdf = PdfReader(file)
        text = "".join(page.extract_text() or "" for page in pdf.pages)

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file.name}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({"source": file.name})

    # Generate embeddings
    all_embeddings = [get_embedding(chunk) for chunk in all_chunks]

    # Store in ChromaDB
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadata,
        embeddings=all_embeddings
    )

    st.success(f"✅ Added {len(all_chunks)} chunks to database.")


# ---------------------------
# Query Section
# ---------------------------

query = st.text_input("Ask a question from your uploaded documents:")

if query:

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    retrieved_docs = results["documents"][0]

    if not retrieved_docs:
        st.warning("No relevant documents found.")
        st.stop()

    context = "\n\n".join(retrieved_docs)

    # Improved prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the question strictly using ONLY the context provided.
If the answer is not present in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt   # Correct for generation
    )

    st.subheader("📌 Retrieved Context (for verification)")
    st.write(context)

    st.subheader("🤖 Answer")
    st.write(response.text)