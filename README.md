# CourseMate AI

CourseMate AI is a dark-themed PDF study companion. Upload a book, ask questions about it, and follow each answer back to the relevant pages.

## Features

- Upload text-based PDF books up to 50 MB.
- Search uploaded books with maximum marginal relevance (MMR).
- Get answers grounded in the book, with supporting page excerpts.
- Keep multiple chat threads and reopen earlier conversations during the current session.
- Keep multiple books on a session-only bookshelf.
- Build vector indexes in memory; CourseMate does not create a Chroma database or save an index to disk.

The PDF is written to a temporary file only while its pages are extracted. The vector index stays in the running app's memory and is cleared when the browser session or app process ends. Upload and index the PDF again after restarting the app.

## Requirements

- Python 3.12
- A Groq API key to ask questions. PDF indexing works without a key.

## Setup

From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the Groq key in a local `.env` file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder value for `GROQ_API_KEY`. You can also configure the key with Streamlit secrets. The actual `.env` and `.streamlit/secrets.toml` files are ignored by Git.

For Streamlit secrets, create `.streamlit/secrets.toml` with:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

## Start the app

```powershell
streamlit run streamlit_app.py
```

Upload a PDF and choose **Index this book**. The first indexing run downloads the local embedding model and can take a few minutes. When indexing finishes, open a suggested prompt or type a question in the chat box. Previously indexed PDFs remain available on the bookshelf for the current session.

## Command-line mode

The original command-line workflow uses the bundled deep-learning PDF and builds its index in memory when it starts:

```powershell
python main.py
```

To build the index and print its page and passage counts without starting a chat:

```powershell
python create_database.py
```

## Project layout

```text
.
├── streamlit_app.py          # Upload, session bookshelf, and chat interface
├── create_database.py        # PDF parsing, chunking, and in-memory indexing
├── main.py                   # Shared retrieval and answer generation
├── document loaders/         # Sample PDFs and document-loader examples
├── retrievers/               # Retrieval experiments
└── vector store/             # In-memory vector-store example
```
