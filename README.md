# CourseMate AI

CourseMate AI is a command-line retrieval-augmented generation (RAG) assistant. It indexes the supplied deep-learning PDF in ChromaDB, retrieves relevant passages with maximum marginal relevance (MMR), and uses a Groq-hosted language model to answer questions from those passages.

## Requirements

- Python 3.12
- A Groq API key

## Setup

From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file from the example and add your Groq API key:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder value for `GROQ_API_KEY`. The `.env` file is ignored by Git; do not commit it.

## Build the document index

The included `document loaders/deeplearning.pdf` is used as the source document. Create the ChromaDB index with:

```powershell
python create_database.py
```

This creates a local `chroma_db/` directory. The first run also downloads the `sentence-transformers/all-MiniLM-L6-v2` embedding model.

## Ask questions

Start the assistant after the index has been created:

```powershell
python main.py
```

Type a question about the indexed document. Enter `0` to exit. Answers are grounded in the retrieved document passages; when the passages do not contain an answer, the assistant says it could not find one in the document.

## Other examples

The `retrievers/` directory contains standalone examples for MMR, multi-query retrieval, and arXiv search. `vector store/DB.py` and `document loaders/` contain additional experiments with vector stores and document loaders.

## Project layout

```text
.
├── create_database.py       # Loads the PDF, splits it, and creates the ChromaDB index
├── main.py                  # Interactive RAG assistant
├── document loaders/        # Source PDFs and document-loader examples
├── retrievers/              # Retrieval examples
└── vector store/            # Vector-store example
```

The `.env` file, Python environment, generated ChromaDB data, and local model cache are excluded from version control.
