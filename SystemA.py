from __future__ import annotations
import os
import uuid
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Inventory Chatbot", layout="wide")
DATA_PATH = Path("data/inventory.csv")
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"  # หรือ "gemini-1.5-flash" ก็ได้


# ------------------- API KEY -------------------
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state["GOOGLE_API_KEY"] = ""

st.sidebar.title("🔐 API Key")
st.session_state["GOOGLE_API_KEY"] = st.sidebar.text_input(
    "Enter your Google API Key:", type="password", value=st.session_state["GOOGLE_API_KEY"]
)

if not st.session_state["GOOGLE_API_KEY"]:
    st.warning("Please enter your Google API Key to continue.")
    st.stop()

genai.configure(api_key=st.session_state["GOOGLE_API_KEY"])

# ------------------- LOAD INVENTORY -------------------
@st.cache_data
def load_inventory(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df

df = load_inventory(DATA_PATH)
K_RETRIEVE = len(df)
# ------------------- EMBEDDING -------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    return np.vstack([
        genai.embed_content(
            model=EMBED_MODEL,
            content=t,
            task_type="retrieval_document"
        )["embedding"]
        for t in texts
    ])

def create_corpus(df: pd.DataFrame) -> list[str]:
    return [
        f"Item {row.ItemID}: {row.ItemName}, Category: {row.Category}, Type: {row.ItemType}, "
        f"Stock: {row.QuantityInStock} {row.Unit}, Cost: {row.UnitCost} USD, "
        f"Reorder Point: {row.ReorderPoint}, Lead time is {row.LeadTimeDays} days, "
        f"Location: {row.Location}, Last received on {row.LastReceived}"
        for _, row in df.iterrows()
    ]

corpus = create_corpus(df)
embeddings = embed_texts(corpus)

# ------------------- BUILD FAISS -------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ------------------- GEMINI CHAT -------------------
def query_inventory(question: str) -> str:
    # ฝังคำถาม
    q_embed = embed_texts([question])
    scores, indices = index.search(q_embed, K_RETRIEVE)
    snippets = [corpus[i] for i in indices[0]]

    prompt = f"""You are a helpful office inventory assistant.
Answer ONLY based on the following data snippets, do not guess numbers.
Snippets:
{chr(10).join(f"- {s}" for s in snippets)}

Question: {question}
Answer:"""

    chat = genai.GenerativeModel(model_name=CHAT_MODEL).start_chat()
    reply = chat.send_message(prompt)
    return reply.text

# ------------------- UI -------------------
st.title("📦 Inventory RAG Chatbot")
st.write("Ask me anything about the inventory 👇")

query = st.text_input("Your Question:")
if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        answer = query_inventory(query)
        st.markdown("**Answer:**")
        st.success(answer)

# ------------------- OPTIONAL: Show table -------------------
with st.expander("📊 View Inventory Data"):
    st.dataframe(df)