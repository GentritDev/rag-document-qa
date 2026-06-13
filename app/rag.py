"""
Core RAG (Retrieval-Augmented Generation) logic.

Pipeline:
    1. Load a PDF document
    2. Split it into overlapping text chunks (tagged with their source filename)
    3. Embed chunks locally with a HuggingFace sentence-transformer (free, CPU)
    4. Store / retrieve embeddings via ChromaDB (persisted to disk)
    5. Generate an answer using Gemini 2.0 Flash (Google AI Studio free tier)

The RAGEngine class is designed to be instantiated once (singleton via
`get_rag_engine`) and reused across requests, since loading the embedding
model is the most expensive part of startup.

Indexing supports an optional progress callback so the UI can show
fine-grained progress (loading -> splitting -> embedding in batches).

Retrieval supports an optional `source_filter` so questions can be scoped
to a single uploaded document, or searched across all of them.
"""

from pathlib import Path
from typing import Callable, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import settings

# Sentinel value used by the UI/API to mean "search across all documents".
ALL_DOCUMENTS = "All documents"

# A progress callback receives (fraction_complete, description).
ProgressCallback = Callable[[float, str], None]


class RAGEngineError(Exception):
    """Base exception for RAG engine failures."""


class DocumentLoadError(RAGEngineError):
    """Raised when a document cannot be loaded or parsed."""


class RetrievalError(RAGEngineError):
    """Raised when retrieval or generation fails."""


class RAGEngine:
    """
    Encapsulates the full RAG pipeline: indexing documents and
    answering questions against the indexed vector store.
    """

    def __init__(self) -> None:
        logger.info("Initializing embedding model: {}", settings.embedding_model)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            logger.exception("Failed to load embedding model")
            raise RAGEngineError("Could not initialize embedding model") from exc

        logger.info("Initializing Gemini LLM: {}", settings.llm_model)
        try:
            self.llm = ChatGroq(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                groq_api_key=settings.groq_api_key,
            )
        except Exception as exc:
            logger.exception("Failed to initialize Gemini client")
            raise RAGEngineError("Could not initialize LLM client") from exc

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        self.vectorstore: Optional[Chroma] = None
        self.indexed_files: set[str] = set()
        self._try_load_existing_vectorstore()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _try_load_existing_vectorstore(self) -> None:
        """Load a previously persisted ChromaDB collection, if one exists."""
        persist_dir = settings.vectorstore_dir
        has_existing_data = persist_dir.exists() and any(persist_dir.iterdir())

        if not has_existing_data:
            return

        try:
            self.vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
            )
            self._refresh_indexed_files()
            logger.info(
                "Loaded existing vector store from {} ({} document(s))",
                persist_dir,
                len(self.indexed_files),
            )
        except Exception:
            logger.warning(
                "Found a vectorstore directory but failed to load it. "
                "It will be recreated on the next document upload."
            )

    def _refresh_indexed_files(self) -> None:
        """Rebuild the set of known source filenames from vector store metadata."""
        if self.vectorstore is None:
            self.indexed_files = set()
            return

        try:
            records = self.vectorstore.get()
            metadatas = records.get("metadatas") or []
            self.indexed_files = {
                m["source"] for m in metadatas if m and m.get("source")
            }
        except Exception:
            logger.debug("Could not read metadata from vector store", exc_info=True)

    @property
    def is_ready(self) -> bool:
        """Whether a vector store is loaded and ready to answer questions."""
        return self.vectorstore is not None

    def get_indexed_files(self) -> list[str]:
        """Return a sorted list of filenames currently indexed."""
        return sorted(self.indexed_files)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_document(
        self,
        file_path: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> int:
        """
        Load a PDF, split it into chunks, embed (in batches) and persist to ChromaDB.

        Args:
            file_path: Path to the PDF file on disk.
            progress_callback: Optional callback(fraction, description) invoked
                as indexing progresses, for UI progress bars.

        Returns:
            Number of chunks indexed.

        Raises:
            DocumentLoadError: If the file cannot be read or parsed,
                or contains no extractable text.
            RAGEngineError: If embedding or storage fails.
        """

        def report(fraction: float, description: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(min(fraction, 1.0), description)
            except Exception:
                logger.debug("Progress callback raised, ignoring", exc_info=True)

        if not file_path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")

        report(0.05, f"Loading {file_path.name}...")
        logger.info("Loading document: {}", file_path.name)
        try:
            loader = PyPDFLoader(str(file_path))
            documents: list[Document] = loader.load()
        except Exception as exc:
            logger.exception("Failed to load PDF: {}", file_path.name)
            raise DocumentLoadError(f"Could not parse PDF '{file_path.name}'") from exc

        if not documents:
            raise DocumentLoadError(f"No extractable text found in '{file_path.name}'")

        report(0.2, f"Splitting {len(documents)} page(s) into chunks...")
        chunks = self.text_splitter.split_documents(documents)

        if not chunks:
            raise DocumentLoadError(f"No text chunks produced from '{file_path.name}'")

        # Tag every chunk with its source filename so questions can later
        # be scoped to this document.
        for chunk in chunks:
            chunk.metadata["source"] = file_path.name

        logger.info("Split '{}' into {} chunks", file_path.name, len(chunks))

        try:
            if self.vectorstore is None:
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=str(settings.vectorstore_dir),
                )

            batch_size = max(settings.embedding_batch_size, 1)
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            for batch_index in range(total_batches):
                start = batch_index * batch_size
                end = min(start + batch_size, len(chunks))
                batch = chunks[start:end]

                self.vectorstore.add_documents(batch)

                fraction = 0.3 + 0.65 * ((batch_index + 1) / total_batches)
                report(
                    fraction,
                    f"Embedding chunks {end}/{len(chunks)}...",
                )
        except Exception as exc:
            logger.exception("Failed to add documents to vector store")
            raise RAGEngineError("Embedding/storage step failed") from exc

        self.indexed_files.add(file_path.name)
        report(1.0, "Done")
        logger.info("Indexing complete for {}", file_path.name)
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_relevant_chunks(
        self,
        question: str,
        source_filter: Optional[str] = None,
    ) -> list[Document]:
        """
        Retrieve the top-k chunks relevant to `question`.

        Args:
            question: The user's natural-language question.
            source_filter: If set (and not ALL_DOCUMENTS), restrict retrieval
                to chunks whose `source` metadata matches this filename.

        Raises:
            RetrievalError: If no documents are indexed, the question is
                empty, or the underlying retriever call fails.
        """
        if self.vectorstore is None:
            raise RetrievalError("No documents have been indexed yet. Upload a PDF first.")

        if not question.strip():
            raise RetrievalError("Question cannot be empty.")

        search_kwargs: dict = {"k": settings.retrieval_k}
        if source_filter and source_filter != ALL_DOCUMENTS:
            search_kwargs["filter"] = {"source": source_filter}

        try:
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            return retriever.invoke(question)
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise RetrievalError("Failed to retrieve relevant documents") from exc

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_answer(self, question: str, relevant_docs: list[Document]) -> str:
        """
        Generate an answer to `question` grounded in `relevant_docs`.

        Raises:
            RetrievalError: If the LLM call fails.
        """
        if not relevant_docs:
            return "No relevant information found in the indexed documents."

        context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

        prompt = (
            "You are a helpful technical documentation assistant. "
            "Answer the question using ONLY the context below. "
            "If the answer is not contained in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as exc:
            logger.exception("LLM generation failed")
            raise RetrievalError("Failed to generate an answer from the LLM") from exc

    # ------------------------------------------------------------------
    # Convenience: retrieve + generate in one call (used by the REST API)
    # ------------------------------------------------------------------
    def answer_question(
        self,
        question: str,
        source_filter: Optional[str] = None,
    ) -> dict:
        """
        Retrieve relevant chunks and generate an answer.

        Returns:
            A dict with keys:
                - "answer": the generated answer (str)
                - "sources": list of source text snippets used (list[str])
        """
        relevant_docs = self.retrieve_relevant_chunks(question, source_filter)
        answer_text = self.generate_answer(question, relevant_docs)
        sources = [doc.page_content[:200].strip() + "..." for doc in relevant_docs]

        logger.info("Answered question using {} source chunk(s)", len(sources))
        return {"answer": answer_text, "sources": sources}


# ----------------------------------------------------------------------
# Singleton accessor
# ----------------------------------------------------------------------
_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """
    Return a shared, lazily-initialized RAGEngine instance.

    Avoids reloading the embedding model and recreating the LLM client
    on every request.
    """
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
