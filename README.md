# Project 1 — Company RAG Chatbot (Setup Guide)

## Folder structure
```
project1_rag_chatbot/
├── data/                 # put your company .txt / .pdf / .docx files here
├── ingest.py             # builds the vector database from /data
├── app.py                # Streamlit GUI chatbot
├── requirements.txt
├── .env.example
└── vectorstore/          # auto-created after you run ingest.py
```

## Step-by-step setup in VS Code

### 1. Open the folder
- Open VS Code → File → Open Folder → select `project1_rag_chatbot`.

### 2. Create a virtual environment
Open the VS Code terminal (`Ctrl + ~`) and run:
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

In VS Code, also select this venv as your Python interpreter:
`Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `venv`.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your free Groq API key
This project uses **free, local embeddings** (no key needed) for search, and
**Groq** (free tier, very fast) for the chat responses.
- Get a free key at https://console.groq.com/keys (just sign up, no card needed)
- Copy `.env.example` and rename the copy to `.env`
- Open `.env` and paste your real key:
```
GROQ_API_KEY=gsk-xxxxxxxxxxxxxxxx
```

### 5. Add your company data
Replace/add files inside the `data/` folder — `.txt`, `.pdf`, or `.docx` all work.
A sample file `company_info.txt` is already there so you can test immediately.

### 6. Build the knowledge base
```bash
python ingest.py
```
This reads everything in `/data`, chunks it, creates embeddings, and saves a
`vectorstore/` folder (your local vector database).
*(First run downloads a small local embedding model, ~90MB — one-time, needs internet.)*

### 7. Launch the chatbot GUI
```bash
streamlit run app.py
```
This opens a browser window (usually `http://localhost:8501`) with a full
chat interface: message bubbles, chat history, source citations, and a
clear-chat button in the sidebar.

## Updating the data later
Whenever you add/change files in `/data`, just re-run:
```bash
python ingest.py
```
then restart the Streamlit app.

## Notes
- **Embeddings** run locally on your machine (`sentence-transformers/all-MiniLM-L6-v2`)
  — 100% free, no API key, no cost, but the first run downloads the ~90MB model.
- **Chat responses** use Groq's free tier (`llama-3.1-8b-instant`) — fast and
  free, but has a rate limit on the free plan (fine for testing/dev use).
- Everything (vector index) is stored **locally** in `/vectorstore` — no
  external database needed.
- If you ever want to go fully offline (no internet at all), the LLM part
  can be swapped to run locally too via Ollama — just ask.
