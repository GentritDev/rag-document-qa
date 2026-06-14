---
title: Tech Documentation Assistant
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: frontend/app.py
pinned: false
---

#  RAG Tech Documentation Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system for intelligent document Q&A. Upload technical PDFs and ask natural language questions to get contextually accurate answers powered by LLMs.

**[Live Demo](https://huggingface.co/spaces/Genti123/rag-document-qa)** 
---
 [API Docs](#-rest-api-endpoints) |  [Docker](#docker-deployment)

---

##  Key Features

- **PDF Document Indexing**: Upload and automatically index technical documents
- **Semantic Search**: Retrieve relevant content using embeddings and ChromaDB
- **Intelligent Q&A**: Generate contextually accurate answers using Groq LLM (Llama 3.3 70B)
- **Document Scoping**: Ask questions across all documents or scope to specific files
- **REST API**: Full REST API for programmatic access (`/upload`, `/ask`, `/health`)
- **Interactive UI**: Gradio-based frontend for user-friendly interaction
- **Local Embeddings**: Free, CPU-friendly HuggingFace embeddings (no external API calls for embeddings)
- **Persistent Vector Store**: ChromaDB for efficient document retrieval across sessions

---

##  Architecture

### Core Pipeline

```
PDF Upload
    ↓
[PyPDFLoader] - Parse PDF pages
    ↓
[RecursiveCharacterTextSplitter] - Split into 500-char overlapping chunks
    ↓
[HuggingFaceEmbeddings] - Generate embeddings (all-MiniLM-L6-v2)
    ↓
[ChromaDB] - Store & index embeddings persistently
    ↓
Query Processing
    ↓
[Semantic Search] - Retrieve top-3 relevant chunks
    ↓
[Prompt Engineering] - Construct grounded context
    ↓
[ChatGroq LLM] - Generate answer from context
    ↓
Response with Sources
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend Framework** | FastAPI | High-performance REST API |
| **LLM** | Groq API (Llama 3.3 70B) | Fast, free-tier LLM inference |
| **Embeddings** | HuggingFace Transformers | Local, CPU-friendly semantic embeddings |
| **Vector DB** | ChromaDB | Persistent vector store for retrieval |
| **Document Processing** | PyPDF, LangChain | PDF parsing and text handling |
| **Frontend** | Gradio | Interactive web UI |
| **Server** | Uvicorn | ASGI server |
| **Configuration** | Pydantic Settings | Environment-based config |
| **Logging** | Loguru | Structured logging |

---

##  Getting Started

### Prerequisites
- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Local Installation

```bash
# Clone repository
git clone https://github.com/GentritDev/rag-document-qa.git
cd rag-document-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Run application
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Visit `http://localhost:7860` for the UI or `http://localhost:7860/docs` for API documentation.

### Docker Deployment

```bash
# Build image
docker build -t rag-doc-qa .

# Run container
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_groq_api_key_here \
  -v $(pwd)/data:/code/data \
  -v $(pwd)/vectorstore:/code/vectorstore \
  rag-doc-qa
```

### Hugging Face Spaces

Push to Hugging Face Spaces:
```bash
huggingface-cli repo create rag-document-qa
git remote add huggingface https://huggingface.co/spaces/YourUsername/rag-document-qa
git push huggingface main
```

---

##  REST API Endpoints

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "ok",
  "vectorstore_ready": true
}
```

### Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

# Body: file=<pdf_file>
```
Response:
```json
{
  "filename": "technical_docs.pdf",
  "chunks_indexed": 42,
  "indexed_files": ["technical_docs.pdf"],
  "message": "Document indexed successfully."
}
```

### List Documents
```bash
GET /documents
```
Response:
```json
{
  "indexed_files": ["technical_docs.pdf", "api_reference.pdf"]
}
```

### Ask Question
```bash
POST /ask
Content-Type: application/json

{
  "question": "How do I authenticate with the API?",
  "source": "api_reference.pdf"  // Optional: leave null to search all
}
```
Response:
```json
{
  "answer": "Authentication is handled via OAuth 2.0...",
  "sources": [
    "Authentication uses OAuth 2.0 with bearer tokens...",
    "For API keys, request one from the dashboard..."
  ]
}
```

### API Documentation
Full interactive documentation: `http://localhost:7860/docs`

---

##  Configuration

Configure via environment variables or `.env` file:

```env
# LLM Configuration
GROQ_API_KEY=your_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Storage
VECTORSTORE_DIR=./vectorstore
DATA_DIR=./data

# Text Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=3

# Upload Limits
MAX_FILE_SIZE_MB=10

# Performance
EMBEDDING_BATCH_SIZE=16
LOG_LEVEL=INFO
```

---

##  Project Structure

```
rag-document-qa/
├── app/
│   ├── main.py              # FastAPI app, endpoint definitions
│   ├── rag.py               # Core RAG engine, indexing & retrieval
│   ├── config.py            # Configuration management
│   └── __init__.py
├── frontend/
│   └── ui.py               # Gradio UI
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker containerization
├── .env.example             # Example environment variables
└── README.md                # This file
```

---

## How It Works

### Indexing Phase (Document Upload)

1. **PDF Parsing**: PyPDFLoader extracts text from uploaded PDF
2. **Chunking**: RecursiveCharacterTextSplitter creates 500-char overlapping chunks
3. **Embedding**: HuggingFace model generates embeddings for each chunk (~384 dims)
4. **Storage**: ChromaDB stores embeddings with metadata (filename, page number)
5. **Persistence**: Vector store saved to disk for session continuity

### Retrieval Phase (Question Answering)

1. **Query Embedding**: User question converted to embedding using same model
2. **Semantic Search**: ChromaDB finds top-3 most similar chunks
3. **Context Construction**: Relevant chunks combined into a context string
4. **Prompt Engineering**: Context + question formatted for LLM
5. **LLM Generation**: Groq API processes prompt, generates grounded answer
6. **Source Attribution**: Returns answer with source snippet references

---

## Key Design Decisions

- **Local Embeddings**: Reduces latency, no per-embedding API cost
- **CPU-Friendly Model**: all-MiniLM-L6-v2 optimized for inference speed
- **Batch Embedding**: Processes chunks in batches for memory efficiency
- **ChromaDB**: Lightweight, serverless vector DB with disk persistence
- **Singleton Pattern**: RAGEngine instantiated once; shared across requests
- **Structured Logging**: Loguru provides detailed debugging without verbosity
- **Environment-Based Config**: Runs on local machine, Docker, and Hugging Face Spaces without code changes

---

##  Performance Characteristics

- **Embedding Time**: ~5-15s per 10-page PDF (CPU-dependent)
- **Query Latency**: ~1-3s (search + LLM generation)
- **Memory Footprint**: ~2-3GB (model + vector store)
- **Throughput**: Suitable for concurrent requests (FastAPI + async)

---

##  Error Handling

The system includes comprehensive error handling:

- **DocumentLoadError**: Invalid/corrupt PDF, no extractable text
- **RAGEngineError**: Model loading, LLM initialization failures
- **RetrievalError**: No documents indexed, query failures
- HTTP status codes: 400 (client error), 413 (file too large), 500 (server error)
---

##  Dependencies

See `requirements.txt` for complete list. Key packages:

- **langchain** (0.3.7): LLM orchestration
- **langchain-groq** (0.2.1): Groq API integration
- **chromadb** (0.5.18): Vector storage
- **sentence-transformers** (3.3.1): Embedding model
- **fastapi** (0.115.5): REST framework
- **uvicorn** (0.32.0): ASGI server
- **gradio**: UI framework

---

##  Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'gradio'`
**Solution**: Ensure `gradio` is installed in requirements.txt
```bash
pip install gradio --upgrade
```

### Issue: Slow embedding on first run
**Solution**: First request loads the embedding model (~2GB). Subsequent requests use cached model.

### Issue: `GROQ_API_KEY` not found
**Solution**: Create `.env` file or set environment variable:
```bash
export GROQ_API_KEY=your_key_here
```

### Issue: ChromaDB persistence errors
**Solution**: Ensure vectorstore directory is writable:
```bash
mkdir -p ./vectorstore ./data
chmod 755 ./vectorstore ./data
```

---

##  References & Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Groq API](https://console.groq.com/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://gradio.app/)

---

##  License

MIT License - Feel free to use for personal and commercial projects.

---

##  Author

**Gentrit Dev**  
GitHub: [@GentritDev](https://github.com/GentritDev)  
Live Demo: https://huggingface.co/spaces/Genti123/rag-document-qa

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for technical documentation Q&A**
