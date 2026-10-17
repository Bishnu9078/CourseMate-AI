# CourseMate AI

CourseMate AI is a dark-themed, RAG-based PDF study companion. It indexes uploaded books, retrieves relevant passages for each question, and uses those passages to generate answers with page references.

## Features

- Upload text-based PDF books up to 50 MB.
- Search uploaded books with maximum marginal relevance (MMR).
- Get answers grounded in the book, with supporting page excerpts.
- Generate page-cited flashcards and multiple-choice quizzes from a book.
- Track learned flashcards and save quiz scores separately for each chat.
- Keep multiple books and chat threads across app restarts.
- Use the refreshed Streamlit reading-room interface for books, chats, and study tools.
- Start each new session with a blank chat view; reopen saved conversations independently from the sidebar.
- Build vector indexes in memory and rebuild them from saved PDFs when needed.

## Technologies used

- Python 3.12.
- Streamlit for the web interface.
- A Groq API key for answers, flashcards, and quizzes. PDF indexing works without a key.
- LangChain `ChatGroq` with the `openai/gpt-oss-120b` model for generation.
- RAG workflow: `create_database.py` loads, chunks, embeds, and indexes PDF text; `main.py` retrieves relevant passages with MMR and sends them with the question to the model for a book-grounded answer.
- LangChain `PyPDFLoader` and `RecursiveCharacterTextSplitter` to load PDFs and create searchable passages.
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` for local text embeddings.
- LangChain `InMemoryVectorStore` and maximum marginal relevance (MMR) retrieval to find relevant passages.
- SQLite (`sqlite3`) for local books, PDFs, chats, notes, and study progress.
- `python-dotenv` and Streamlit secrets for API key configuration.

Books, PDFs, chats, and study progress are saved locally in the ignored `.coursemate/` directory. A temporary working copy is used during page extraction; vector indexes stay in memory and are rebuilt from the saved PDFs after an app restart. This local store is shared by sessions using the same app installation.

## Requirements

- Python 3.12
- A Groq API key to ask questions. PDF indexing works without a key.

## Project workflow

```mermaid
flowchart TD
    A([Start]) --> B[Upload a text-based PDF]
    B --> C{PDF is valid and readable?}
    C -- No --> D[Show an error and choose another PDF]
    D --> B
    C -- Yes --> E[Extract page text]
    E --> F[Split text into overlapping passages]
    F --> G[Create local embeddings]
    G --> H[Build an in-memory vector index]
    G --> I[Save PDF and book details locally]
    H --> J{Choose an activity}

    J -- Ask a question --> K[Find relevant passages with MMR]
    K --> L[Send passages and question to the language model]
    L --> M[Show a grounded answer with page references]
    M --> J

    J -- Study --> N[Generate flashcards or a quiz]
    N --> O[Save conversations, study materials, and progress locally]
    O --> J

    I --> P[Restart the app]
    P --> Q[Rebuild the in-memory index from the saved PDF]
    Q --> J

    classDef start fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef process fill:#f1f5f9,stroke:#64748b,color:#0f172a
    classDef decision fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef result fill:#dcfce7,stroke:#16a34a,color:#14532d
    class A start
    class C,J decision
    class M,O result
    class B,D,E,F,G,H,I,K,L,N,P,Q process
```

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

Upload a PDF and choose **Index this book**. The first indexing run downloads the local embedding model and can take a few minutes. When indexing finishes, ask questions or generate flashcards and a quiz. The bookshelf, conversations, and study progress remain available after restarting the app.

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
├── streamlit_app.py          # Upload, saved bookshelf, and study interface
├── create_database.py        # PDF parsing, chunking, and in-memory indexing
├── coursemate_store.py       # Local SQLite persistence for books and study progress
├── study_mode.py             # Book-grounded flashcard and quiz generation
├── main.py                   # Shared retrieval and answer generation
├── document loaders/         # Sample PDFs and document-loader examples
├── retrievers/               # Retrieval experiments
└── vector store/             # In-memory vector-store example
```
