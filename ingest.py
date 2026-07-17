"""
ingest.py
Reads every file in ./data, splits it into chunks, embeds it,
and saves a local FAISS vector store to ./vectorstore

Run this ONCE (and again whenever you update your company documents):
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"


def load_documents():
    docs = []

    loaders = [
        DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(
            DATA_DIR, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader
        ),
    ]

    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Skipping a loader due to error: {e}")

    return docs


def main():
    print("Loading documents from ./data ...")
    documents = load_documents()

    if not documents:
        print("No documents found in ./data. Add .txt, .pdf, or .docx files and re-run.")
        return

    print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunk(s).")

    print("Creating embeddings and building FAISS index...")
    print("(First run downloads the local embedding model, ~90MB, one time only)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Done. Vector store saved to ./{VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
