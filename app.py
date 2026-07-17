"""
app.py
Streamlit GUI for the company RAG chatbot.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTORSTORE_DIR = "vectorstore"

st.set_page_config(page_title="Company AI Assistant", page_icon="🤖", layout="centered")


@st.cache_resource(show_spinner=False)
def load_retriever():
    if not os.path.exists(VECTORSTORE_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)


def build_prompt(question, context, chat_history):
    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    return f"""You are a helpful company assistant. Answer the user's question
using ONLY the context below. If the answer isn't in the context, say you
don't have that information.

Context:
{context}

Conversation so far:
{history_text}

Question: {question}
Answer:"""


st.title("🤖 Company AI Assistant")
st.caption("Ask me anything about the company documents you loaded into /data")

retriever = load_retriever()

if retriever is None:
    st.error(
        "No vector store found. Run `python ingest.py` first to process your "
        "company documents, then restart this app."
    )
    st.stop()

llm = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about the company."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(user_input)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = build_prompt(user_input, context, st.session_state.messages[:-1])
            response = llm.invoke(prompt)
            answer = response.content

            st.markdown(answer)

            if docs:
                with st.expander("📄 Sources used"):
                    for i, doc in enumerate(docs, 1):
                        src = doc.metadata.get("source", "unknown")
                        st.markdown(f"**{i}. {src}**")
                        st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me anything about the company."}
        ]
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Documents in `/data` are chunked & embedded\n"
        "2. Stored in a local FAISS vector index\n"
        "3. Your question retrieves the most relevant chunks\n"
        "4. GPT answers using only that retrieved context"
    )
