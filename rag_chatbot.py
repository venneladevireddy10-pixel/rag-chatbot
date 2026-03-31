# ========================================
# STREAMLIT RAG SYSTEM Chatbot
# ========================================

import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------

st.set_page_config(page_title = "RAG Chatbot", layout = "centered")

st.title("🧠 RAG Chatbot")
st.write("Ask questions about AI concepts.")

# -----------------------------
# Knowledge Base
# -----------------------------


documents = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning is a branch of machine learning that uses neutral networks with multiple layers.",
    "Natural Language Processing is a field of artificial intelligence focused on understandung and processing human lanaguage.",
    "Large Language Models aer powerful AI systems trained on massive text datasets.",
    "Reteival Augumented Generation combines information retrieval with text generation models."
]

# -----------------------------
# Load Model (Load Only Once)
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Create embeddings (cache them)
@st.cache_data
def create_embeddings(docs):
    return model.encode(docs)

document_embeddings = create_embeddings(documents)

# -----------------------------
# RAG Function
# -----------------------------

def rag_system(query):
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, document_embeddings)
    best_match_index = np.argmax(similarity_scores)
    return documents[best_match_index]

# ---------------------------
# Chat Interface
# ---------------------------

user_input = st.text_input("Ask a question : ")

if st.button("Get Answer"):
    if user_input.strip() != "":
        answer = rag_system(user_input)
        st.success("Answer : ")
        st.write(answer)
    else:
        st.warning("Please enter a question.")