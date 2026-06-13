# Tech Documentation Assistant — RAG-Powered Q&A

A Retrieval-Augmented Generation (RAG) application that lets you upload a
technical PDF (API reference, internal documentation, cloud service docs,
etc.) and ask natural-language questions about it, with answers grounded
in the document's actual content.

Built entirely on a **free, open-source stack** — no paid API keys required.

---

## Problem

Generic LLMs don't know anything about your internal or proprietary
documents, and they can hallucinate when asked about specifics. This
project solves that by combining:

1. **Retrieval** — find the most relevant passages from a document
2. **Augmentation** — inject those passages into the LLM's prompt as context
3. **Generation** — let the LLM answer using only that grounded context

This is the same pattern used in production "chat with your docs" tools,
internal knowledge bases, and customer support assistants.

---

## Architecture

```
┌──────────────┐      ┌────────────────────┐      ┌──────────────────┐
│   PDF file    │ ───▶ │  Text Splitter      │ ───▶ │  HuggingFace      │
│  (uploaded)   │      │  (chunking)         │      │  Embeddings (CPU) │
└──────────────┘      └────────────────────┘      └─────────┬────────┘
                                                              ▼
                                                     ┌──────────────────┐
                                                     │     ChromaDB      │
                                                     │  (vector store,   │
                                                     │    persisted)     │
                                                     └─────────┬────────┘
                                                              ▼
┌──────────────┐      ┌────────────────────┐      ┌──────────────────┐
│  User question│ ───▶ │  Retriever (top-k)  │ ───▶ │  Llama 3 (Groq)   │
└──────────────┘      └────────────────────┘      │  → grounded answer│
                                                     └──────────────────┘
```

The FastAPI app exposes a small REST API **and** serves an interactive
Gradio UI at the root path, so the whole thing runs as a single service.

---

## Tech Stack

| Component        | Technology                                   | Why                                  |
|-------------------|-----------------------------------------------|----------------------------------------|
| LLM               | **Llama 3 (`llama-3.1-8b-instant`) via Groq** | Free tier, very fast inference          |
| Embeddings        | **sentence-transformers (all-MiniLM-L6-v2)**  | Runs locally on CPU, no API cost        |
| Vector store      | **ChromaDB**                                  | Lightweight, persists to disk           |
| Orchestration     | **LangChain**                                 | Document loading, splitting, chaining   |
| API               | **FastAPI**                                   | Async, typed, auto-generated docs       |
| UI                | **Gradio**                                    | Quick interactive demo, mounted in API  |
| Config            | **Pydantic Settings**                         | Type-safe environment configuration     |
| Logging           | **Loguru**                                    | Structured, readable logs               |
| Deployment        | **Docker**                                    | Single container, portable              |

---

## Project Structure

```
rag-document-qa/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + mounted Gradio UI
│   ├── rag.py            # Core RAG engine (indexing + Q&A)
│   └── config.py         # Pydantic settings
├── frontend/
│   ├── __init__.py
│   └── app.py             # Gradio UI definition
├── data/                  # Uploaded PDFs (gitignored)
├── vectorstore/           # ChromaDB persisted data (gitignored)
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## How to Run

### 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) → API Keys →
Create. The free tier covers `llama-3.1-8b-instant` with generous daily
limits — more than enough for a demo.

### 2. Local setup

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run the app

```bash
uvicorn app.main:app --reload --port 7860
```

- Interactive UI: http://localhost:7860
- API docs (Swagger): http://localhost:7860/docs

### 4. Use it

1. Go to the **Upload Document** tab, select a PDF, click **Index Document**
2. Go to the **Ask Questions** tab and ask something specific to that document
3. The answer and the retrieved source passages are both displayed

---

## API Usage (curl)

```bash
# Health check
curl http://localhost:7860/health

# Upload a document
curl -X POST http://localhost:7860/upload \
  -F "file=@./data/sample.pdf"

# Ask a question
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I authenticate with the API?"}'
```

---

## Running with Docker

```bash
docker build -t rag-doc-assistant .
docker run -p 7860:7860 --env-file .env rag-doc-assistant
```

---

## Deployment

This project is designed to deploy as a **single container** to:

- **Hugging Face Spaces** (Docker SDK) — set `GROQ_API_KEY` as a Space secret
- **Render** (Web Service, Docker) — set `GROQ_API_KEY` as an environment variable
- Any platform that can run a Docker container and exposes port `7860`

---

## Design Notes

- **Singleton RAG engine**: the embedding model and Groq client are
  initialized once at startup (via FastAPI's `lifespan`) and reused across
  requests — avoids reloading a ~90MB model on every API call.
- **Persisted vector store**: ChromaDB writes to disk, so indexed documents
  survive a server restart.
- **Custom exception hierarchy**: `DocumentLoadError` and `RetrievalError`
  are mapped to appropriate HTTP status codes (400 vs 500) instead of
  leaking raw stack traces to clients.
- **Filename sanitization**: uploaded filenames are reduced to their base
  name before being written to disk, preventing path traversal.

---

## Possible Extensions

- Support multiple documents with per-document filtering
- Add streaming responses (`llm.stream(...)`) for token-by-token output
- Swap ChromaDB for a managed vector DB (Pinecone, Qdrant Cloud) for
  multi-user / production scale
- Add conversation memory for follow-up questions

