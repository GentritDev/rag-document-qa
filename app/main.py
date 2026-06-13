"""
FastAPI application for the Tech Documentation Assistant.

Exposes a small REST API (/health, /upload, /ask) for programmatic use,
and mounts the Gradio UI at the root path ("/") so the whole project
runs as a single deployable service (e.g. on Hugging Face Spaces, Render,
or a single Docker container).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from gradio import mount_gradio_app
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.rag import (
    ALL_DOCUMENTS,
    DocumentLoadError,
    RAGEngineError,
    RetrievalError,
    get_rag_engine,
)
from frontend.app import build_demo


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the RAG engine (load embedding model + LLM client) on startup."""
    logger.info("Starting up - initializing RAG engine ...")
    try:
        get_rag_engine()
    except RAGEngineError:
        logger.exception("RAG engine failed to initialize during startup")
        raise
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Tech Documentation Assistant",
    description=(
        "A Retrieval-Augmented Generation (RAG) API for asking questions "
        "about uploaded technical PDF documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    source: Optional[str] = Field(
        default=None,
        description=(
            "Optional filename to scope the search to a single indexed "
            "document. Omit or set to 'All documents' to search everything."
        ),
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    indexed_files: list[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool


class DocumentsResponse(BaseModel):
    indexed_files: list[str]


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Liveness/readiness probe. Reports whether documents are indexed."""
    engine = get_rag_engine()
    return HealthResponse(status="ok", vectorstore_ready=engine.is_ready)


@app.post("/upload", response_model=UploadResponse, tags=["Indexing"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload and index a PDF document.

    The file is saved to the configured data directory, split into
    chunks, embedded locally, and stored in ChromaDB for retrieval.
    """
    is_pdf = (file.content_type == "application/pdf") or (
        file.filename and file.filename.lower().endswith(".pdf")
    )
    if not is_pdf:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # Sanitize filename to avoid path traversal.
    safe_name = Path(file.filename).name
    dest_path = settings.data_dir / safe_name

    try:
        contents = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read uploaded file.",
        ) from exc
    finally:
        await file.close()

    if len(contents) > settings.max_file_size_bytes:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is {size_mb:.1f} MB, which exceeds the "
                f"{settings.max_file_size_mb} MB limit."
            ),
        )

    try:
        dest_path.write_bytes(contents)
    except Exception as exc:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save uploaded file.",
        ) from exc

    engine = get_rag_engine()
    try:
        n_chunks = engine.index_document(dest_path)
    except DocumentLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RAGEngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    return UploadResponse(
        filename=safe_name,
        chunks_indexed=n_chunks,
        indexed_files=engine.get_indexed_files(),
        message="Document indexed successfully.",
    )


@app.get("/documents", response_model=DocumentsResponse, tags=["Indexing"])
async def list_documents() -> DocumentsResponse:
    """List filenames of all currently indexed documents."""
    engine = get_rag_engine()
    return DocumentsResponse(indexed_files=engine.get_indexed_files())


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(payload: AskRequest) -> AskResponse:
    """
    Ask a question about the previously indexed document(s).

    If `source` is provided, the search is scoped to that single document;
    otherwise (or if `source` is 'All documents'), all indexed documents
    are searched.
    """
    engine = get_rag_engine()
    source_filter = payload.source if payload.source != ALL_DOCUMENTS else None

    try:
        result = engine.answer_question(payload.question, source_filter=source_filter)
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except RAGEngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    return AskResponse(**result)


# ----------------------------------------------------------------------
# Mount the Gradio UI at the root path.
# This makes the FastAPI app the single entry point: API docs at /docs,
# interactive UI at /.
# ----------------------------------------------------------------------
demo = build_demo()
app = mount_gradio_app(app, demo, path="/")
