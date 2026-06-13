"""
Gradio UI for the Tech Documentation Assistant.

The UI talks directly to the shared RAGEngine instance (in-process), so
it works both when mounted inside the FastAPI app (production) and when
run standalone for local development (`python -m frontend.app`).

Features:
    - Upload progress bar (loading -> splitting -> embedding in batches)
    - File size limit to protect free/CPU hosting from huge documents
    - Multi-document support with a "search scope" selector
      (defaults to the most recently uploaded document)
    - Automatically follows the user's OS light/dark theme preference
    - Small Terms of Use / Privacy notice
"""

from pathlib import Path

import gradio as gr
from loguru import logger

from app.config import settings
from app.rag import (
    ALL_DOCUMENTS,
    DocumentLoadError,
    RAGEngineError,
    RetrievalError,
    get_rag_engine,
)

# Forces Gradio to follow the OS-level light/dark preference instead of
# always defaulting to light mode. See: https://github.com/gradio-app/gradio
_FOLLOW_SYSTEM_THEME_JS = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'system') {
        url.searchParams.set('__theme', 'system');
        window.location.href = url.href;
    }
}
"""

_TERMS_TEXT = """
**Demo project - please read before uploading.**

- This is a portfolio / demo application, not a production service.
- Uploaded documents are processed locally (chunking + embeddings) and the
  retrieved text passages are sent to a third-party LLM API (Google Gemini)
  to generate answers.
- Do **not** upload confidential, personal, or sensitive documents.
- Uploaded files and indexed data may be cleared at any time without notice.
"""


def _initial_scope_choices_and_value() -> tuple[list[str], str]:
    """Build the initial scope dropdown state from any previously indexed files."""
    engine = get_rag_engine()
    indexed = engine.get_indexed_files()
    choices = [ALL_DOCUMENTS] + indexed
    value = indexed[-1] if indexed else ALL_DOCUMENTS
    return choices, value


def handle_upload(file, progress: gr.Progress = gr.Progress()):
    """
    Validate and index an uploaded PDF.

    Returns:
        (status_message, dropdown_update) - the dropdown update refreshes
        the "search scope" choices and defaults to the newly uploaded file.
    """
    if file is None:
        return "⚠️ Please select a PDF file first.", gr.update()

    source_path = Path(file.name)

    # --- File size guard ---
    try:
        file_size_bytes = source_path.stat().st_size
    except OSError:
        return "❌ Could not read the uploaded file.", gr.update()

    if file_size_bytes > settings.max_file_size_bytes:
        size_mb = file_size_bytes / (1024 * 1024)
        return (
            f"❌ File is {size_mb:.1f} MB, which exceeds the "
            f"{settings.max_file_size_mb} MB limit. Please upload a smaller PDF.",
            gr.update(),
        )

    engine = get_rag_engine()
    dest_path = settings.data_dir / source_path.name

    progress(0, desc=f"Saving {source_path.name}...")
    try:
        dest_path.write_bytes(source_path.read_bytes())
    except OSError as exc:
        logger.exception("Failed to save uploaded file")
        return f"❌ Could not save file: {exc}", gr.update()

    def on_progress(fraction: float, description: str) -> None:
        progress(fraction, desc=description)

    try:
        n_chunks = engine.index_document(dest_path, progress_callback=on_progress)
    except DocumentLoadError as exc:
        logger.warning("Document load error: {}", exc)
        return f"❌ {exc}", gr.update()
    except RAGEngineError as exc:
        logger.exception("Indexing failed")
        return f"❌ Internal error while indexing: {exc}", gr.update()

    choices = [ALL_DOCUMENTS] + engine.get_indexed_files()
    checkmark_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" style="width:20px; height:20px; display:inline-block; vertical-align:middle; color:#10B981; margin-right:5px;">
        <path fill-rule="evenodd" d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12Zm13.36-1.814a.75.75 0 1 0-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 0 0-1.06 1.06l2.5 2.5a.75.75 0 0 0 1.14-.082l3.75-5.25Z" clip-rule="evenodd" />
    </svg>
    """

    status = (
        f"{checkmark_svg}"
        f"<span style='vertical-align:middle;'>Indexed '<b>{dest_path.name}</b>' into {n_chunks} chunks. "
        f"Switch to the 'Ask Questions' tab to query it.</span>"
    )
    return status, gr.update(choices=choices, value=dest_path.name)

def handle_question(
    question: str,
    scope: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    """Answer a question, optionally scoped to a single uploaded document."""
    if not question or not question.strip():
        return "Please enter a question.", ""

    engine = get_rag_engine()
    source_filter = None if scope in (ALL_DOCUMENTS, None, "") else scope

    try:
        progress(0.2, desc="Retrieving relevant passages...")
        relevant_docs = engine.retrieve_relevant_chunks(question, source_filter)

        progress(0.6, desc="Generating answer with Gemini...")
        answer = engine.generate_answer(question, relevant_docs)
        progress(1.0, desc="Done")
    except RetrievalError as exc:
        return f"⚠️ {exc}", ""
    except RAGEngineError as exc:
        logger.exception("Answer generation failed")
        return f"❌ Internal error: {exc}", ""

    if relevant_docs:
        sources_text = "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}] "
            + doc.page_content[:200].strip()
            + "..."
            for doc in relevant_docs
        )
    else:
        sources_text = "No sources retrieved."

    return answer, sources_text


def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks app (not launched)."""
    initial_choices, initial_value = _initial_scope_choices_and_value()

    with gr.Blocks(title="Tech Documentation Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Tech Documentation Assistant

            A Retrieval-Augmented Generation (RAG) demo. Upload a technical PDF
            (API reference, internal wiki export, cloud documentation, etc.)
            and ask questions about it.

            **Stack:** local HuggingFace embeddings (free, CPU) + ChromaDB +
            llama-3.1
            """
        )

        with gr.Tab("1. Upload Document"):
            gr.Markdown(f"Max file size: **{settings.max_file_size_mb} MB**.")
            file_input = gr.File(label="PDF document", file_types=[".pdf"])
            upload_btn = gr.Button("Index Document", variant="primary")
            upload_status = gr.HTML()

        with gr.Tab("2. Ask Questions"):
            scope_dropdown = gr.Dropdown(
                label="Search in",
                choices=initial_choices,
                value=initial_value,
                info="Defaults to the document you just uploaded. "
                "Choose 'All documents' to search everything indexed.",
            )
            question_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. How do I authenticate with the API?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=4, interactive=False)
            sources_output = gr.Textbox(
                label="Retrieved source chunks", lines=6, interactive=False
            )

        # --- Wiring ---
        upload_btn.click(
            handle_upload,
            inputs=file_input,
            outputs=[upload_status, scope_dropdown],
        )

        ask_btn.click(
            handle_question,
            inputs=[question_input, scope_dropdown],
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            handle_question,
            inputs=[question_input, scope_dropdown],
            outputs=[answer_output, sources_output],
        )

        with gr.Accordion("Terms of Use & Privacy", open=False):
            gr.Markdown(_TERMS_TEXT)

        # Follow the visitor's OS light/dark preference.
        demo.load(None, js=_FOLLOW_SYSTEM_THEME_JS)

    return demo


if __name__ == "__main__":
    # Run the UI standalone for local development:
    #   python -m frontend.app
    build_demo().launch()
