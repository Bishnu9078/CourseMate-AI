"""Load PDF books into an in-memory vector store for CourseMate AI."""

from __future__ import annotations

import hashlib
import tempfile
from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "document loaders" / "deeplearning.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the shared local embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _load_and_split_pdf(pdf_path: Path):
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file was not found: {pdf_path}")

    pages = PyPDFLoader(str(pdf_path)).load()
    if not pages:
        raise ValueError("This PDF has no readable pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("No readable text was found in this PDF.")

    return pages, chunks


def create_index_from_pdf(
    pdf_path: str | Path = DEFAULT_PDF_PATH,
) -> tuple[InMemoryVectorStore, dict[str, int]]:
    """Read a PDF and build its searchable index in process memory."""
    pages, chunks = _load_and_split_pdf(Path(pdf_path))
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
    )
    return vectorstore, {"pages": len(pages), "chunks": len(chunks)}


def index_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> tuple[dict[str, str | int], InMemoryVectorStore]:
    """Index an uploaded PDF without retaining its bytes or index on disk."""
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise ValueError("This file does not look like a valid PDF.")

    with tempfile.TemporaryDirectory(prefix="coursemate-upload-") as temporary_directory:
        temporary_pdf = Path(temporary_directory) / "book.pdf"
        temporary_pdf.write_bytes(pdf_bytes)
        vectorstore, stats = create_index_from_pdf(temporary_pdf)

    safe_filename = Path(filename.replace("\\", "/")).name
    metadata: dict[str, str | int] = {
        "fingerprint": hashlib.sha256(pdf_bytes).hexdigest(),
        "filename": safe_filename,
        "pages": stats["pages"],
        "chunks": stats["chunks"],
    }
    return metadata, vectorstore


def create_database(
    pdf_path: str | Path = DEFAULT_PDF_PATH,
) -> tuple[InMemoryVectorStore, dict[str, int]]:
    """Build an in-memory index for the bundled PDF (or a supplied PDF)."""
    return create_index_from_pdf(pdf_path)


if __name__ == "__main__":
    _, result = create_database()
    print(f"Pages loaded: {result['pages']}")
    print(f"Searchable passages created: {result['chunks']}")
    print("The index is in memory for this process and was not saved to disk.")
