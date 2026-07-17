# Company RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a company's internal documents, built with local embeddings, a local vector store, and Groq's free-tier LLM API, wrapped in an interactive Streamlit dashboard.

## Overview

This project tackles a common enterprise use case: letting non-technical staff ask natural-language questions about internal documents (policies, manuals, FAQs, reports) instead of manually searching through files. Rather than sending raw documents to a third-party LLM or relying on an expensive hosted vector database, the entire pipeline runs locally except for the final response generation step.

The goal was to build something that goes beyond a proof-of-concept script: a knowledge base that's built once from a folder of documents, persisted to disk, and served through a proper GUI where anyone can ask a question and get an answer with cited sources — no coding background required.

## Data

The knowledge base is built from whatever files are placed inside the `data/` folder. Supported formats:

| Format | Description |
|---|---|
| `.txt` | Plain text files |
| `.pdf` | PDF documents |
| `.docx` | Word documents |

A sample file, `company_info.txt`, is included so the pipeline can be tested immediately without adding any real company data first.

Each document is split into overlapping chunks, embedded, and stored as vectors so that, at query time, only the most relevant chunks are retrieved and passed to the language model — rather than sending entire documents.

## Pipeline

The project is split into two independent stages, mirroring a typical RAG architecture: **indexing** and **serving**.

**Embeddings** run entirely locally using `sentence-transformers/all-MiniLM-L6-v2` — free, no API key, no per-query cost. The model (~90MB) is downloaded automatically the first time `ingest.py` runs and cached locally afterward, so an internet connection is only required once.

**Chat responses** are generated using Groq's free-tier API (`llama-3.1-8b-instant`), chosen for its speed and zero cost, with the understanding that the free tier carries a rate limit suitable for testing and development rather than high-volume production use.

The vector index itself is stored locally in `vectorstore/` using a lightweight on-disk vector database — no external database service is required.

## How it works

1. `ingest.py` reads every file in `/data`, splits the content into chunks, generates embeddings for each chunk using the local embedding model, and saves the resulting index to `vectorstore/`. This step only needs to be re-run when the contents of `/data` change.
2. `app.py` loads the saved vector index and launches a Streamlit chat interface. When a question is submitted, the app retrieves the most relevant chunks from `vectorstore/`, sends them along with the question to the Groq API, and displays the generated answer alongside the source documents it was based on.

## Project Structure

```
project1_rag_chatbot/
├── data/                 # company .txt / .pdf / .docx files go here
├── ingest.py             # builds the vector database from /data
├── app.py                # Streamlit GUI chatbot
├── requirements.txt
├── .env.example
└── vectorstore/          # generated after running ingest.py (not in repo)
```

## Setup & Usage

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys) (no card required), then copy `.env.example` to `.env` and add it:

```
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxx
```

Add your own files to `data/` (or keep the included sample), then build the knowledge base and launch the app:

```bash
python ingest.py              # builds the vector database (downloads embedding model on first run)
streamlit run app.py          # launches the GUI at http://localhost:8501
```

To update the chatbot's knowledge after changing files in `/data`, re-run `python ingest.py` and restart the Streamlit app.

## Possible Improvements

- Auto-detect changes in `/data` and re-index automatically instead of requiring a manual `ingest.py` run
- Add support for additional file types (e.g. `.md`, `.csv`, `.pptx`)
- Swap the Groq API for a fully local LLM via Ollama to run the entire pipeline offline
- Deploy the Streamlit app for team-wide access (e.g. Streamlit Community Cloud) instead of running it only locally
- Add conversation memory so follow-up questions retain earlier context

## Tech Stack

- **LangChain** (or equivalent RAG framework) — document loading, chunking, and retrieval orchestration
- **Sentence-Transformers** — local embedding generation (`all-MiniLM-L6-v2`)
- **Groq API** — LLM inference (`llama-3.1-8b-instant`)
- **Streamlit** — GUI / chat interface