# app.py

import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df.dropna(subset=["description"], inplace=True)
    return df.reset_index(drop=True)

df = load_data()

# Load models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = pipeline("text-generation", model="sshleifer/tiny-gpt2")
    return embed_model, llm

embed_model, llm = load_models()

# Vectorize descriptions
@st.cache_resource
def create_faiss_index(descriptions):
    embeddings = embed_model.encode(descriptions)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

descriptions = df["description"].tolist()
faiss_index, all_embeddings = create_faiss_index(descriptions)

# Streamlit UI
st.title("🎬 Netflix Movie Recommender with LLM + FAISS")

query = st.text_input("Ask something like 'funny movie after 2015'", "")

if query:
    query_embedding = embed_model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding), k=3)

    st.subheader("🔍 Top Matches:")
    top_matches = [descriptions[i] for i in I[0]]
    for match in top_matches:
        st.write(f"- {match}")

    prompt = f"Here are some Netflix titles based on your query:\n{top_matches}\n\nUser Query: {query}\nPlease give a friendly movie recommendation."
    response = llm(prompt, max_new_tokens=100)[0]['generated_text']

    st.subheader("🤖 LLM Recommendation")
    st.write(response)
