"""Reusable retrieval and answer-generation functions for CourseMate AI."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq

from create_database import DEFAULT_PDF_PATH, create_index_from_pdf


load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"
NO_ANSWER = "I could not find the answer in this book."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are CourseMate, a friendly study assistant.

Use only the supplied book context to answer. Explain ideas clearly and directly.
If the answer is not present in the context, say:
"I could not find the answer in this book."
Do not make up page references or facts.""",
        ),
        (
            "human",
            """Book context:
{context}

Question:
{question}""",
        ),
    ]
)


def load_vectorstore(pdf_path: str | Path = DEFAULT_PDF_PATH) -> InMemoryVectorStore:
    """Build a fresh in-memory vector store from the requested PDF."""
    vectorstore, _ = create_index_from_pdf(pdf_path)
    return vectorstore


def create_retriever(vectorstore: InMemoryVectorStore):
    """Build the diversified MMR retriever used by the app and CLI."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


def retrieve_documents(question: str, retriever) -> list[Document]:
    """Retrieve relevant passages for a question."""
    return retriever.invoke(question)


def build_prompt(question: str, documents: list[Document]):
    """Format retrieved passages into the shared grounding prompt."""
    context = "\n\n".join(document.page_content for document in documents)
    return PROMPT.invoke({"context": context, "question": question})


def get_llm(api_key: str | None = None) -> ChatGroq:
    """Create the Groq chat model, optionally using a per-session API key."""
    options = {"model": MODEL_NAME, "temperature": 0}
    if api_key:
        options["api_key"] = api_key
    return ChatGroq(**options)


def answer_question(
    question: str,
    retriever,
    llm: ChatGroq | None = None,
) -> tuple[str, list[Document]]:
    """Answer a question from retrieved context and return its source passages."""
    documents = retrieve_documents(question, retriever)
    if not documents:
        return NO_ANSWER, documents

    response = (llm or get_llm()).invoke(build_prompt(question, documents))
    return str(response.content), documents


def run_cli() -> None:
    """Run the original interactive command-line assistant."""
    vectorstore = load_vectorstore()
    retriever = create_retriever(vectorstore)
    llm = get_llm()

    print("CourseMate AI is ready. Enter 0 to exit.")
    while True:
        question = input("\nYou: ").strip()
        if question == "0":
            print("Goodbye!")
            break
        if not question:
            continue

        answer, _ = answer_question(question, retriever, llm)
        print(f"\nCourseMate: {answer}")


if __name__ == "__main__":
    run_cli()
