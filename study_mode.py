"""Book-grounded flashcard and quiz generation for CourseMate AI."""

from __future__ import annotations

import json
from json import JSONDecoder
from uuid import uuid4

from langchain_core.vectorstores import VectorStore


LETTERS = ("A", "B", "C", "D")


def _page_number(document) -> int | None:
    page = document.metadata.get("page")
    if isinstance(page, int):
        return page + 1
    if isinstance(page, str) and page.isdigit():
        return int(page) + 1
    return None


def _study_context(vectorstore: VectorStore, query: str) -> tuple[str, dict[str, int | None]]:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    documents = retriever.invoke(query)
    if not documents:
        raise ValueError("No readable passages were found for this study activity.")

    source_pages: dict[str, int | None] = {}
    blocks = []
    for number, document in enumerate(documents, start=1):
        source_id = f"S{number}"
        page = _page_number(document)
        source_pages[source_id] = page
        page_label = f" | page {page}" if page is not None else ""
        blocks.append(
            f"[{source_id}{page_label}]\n{document.page_content.strip()}"
        )
    return "\n\n".join(blocks), source_pages


def _response_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _decode_object(text: str, expected_key: str) -> dict:
    decoder = JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and expected_key in value:
            return value
    raise ValueError("The study response was not valid JSON. Please generate it again.")


def _clean_text(value, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"The generated study material is missing {field} text.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"The generated study material has an empty {field}.")
    return cleaned[:limit]


def _valid_pages(item: dict, source_pages: dict[str, int | None]) -> list[int]:
    source_ids = item.get("source_ids", [])
    if not isinstance(source_ids, list):
        source_ids = []
    pages = {
        source_pages[source_id]
        for source_id in source_ids
        if isinstance(source_id, str)
        and source_id in source_pages
        and source_pages[source_id] is not None
    }
    return sorted(pages)


def generate_flashcards(vectorstore: VectorStore, llm, count: int = 8) -> dict:
    """Generate concise flashcards whose page references come from retrieval."""
    context, source_pages = _study_context(
        vectorstore,
        "important concepts, definitions, methods, and facts useful for study flashcards",
    )
    prompt = f"""Create exactly {count} useful study flashcards from the book context.
Use only facts in the context. Keep each front concise and each back clear.
Return only a JSON object in this schema:
{{"cards":[{{"front":"question or term","back":"answer","source_ids":["S1"]}}]}}
Use one or more source_ids from the context for every card. Do not invent source IDs or page numbers.

BOOK CONTEXT:
{context}
"""
    payload = _decode_object(_response_text(llm.invoke(prompt)), "cards")
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list) or not raw_cards:
        raise ValueError("The model did not return any flashcards. Please generate them again.")

    cards = []
    for item in raw_cards[:count]:
        if not isinstance(item, dict):
            continue
        try:
            cards.append(
                {
                    "front": _clean_text(item.get("front"), "front", 400),
                    "back": _clean_text(item.get("back"), "back", 1200),
                    "pages": _valid_pages(item, source_pages),
                }
            )
        except ValueError:
            continue
    if not cards:
        raise ValueError("The model returned no usable flashcards. Please generate them again.")

    return {
        "id": uuid4().hex,
        "cards": cards,
        "known_indices": [],
    }


def generate_quiz(vectorstore: VectorStore, llm, count: int = 5) -> dict:
    """Generate a multiple-choice quiz grounded in retrieved book passages."""
    context, source_pages = _study_context(
        vectorstore,
        "important concepts and facts that can be assessed with clear multiple-choice questions",
    )
    prompt = f"""Create exactly {count} clear multiple-choice study questions from the book context.
Use only facts in the context. Each question must have four plausible options and one correct answer.
Return only a JSON object in this schema:
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","explanation":"...","source_ids":["S1"]}}]}}
Use one or more source_ids from the context for every question. Do not invent source IDs or page numbers.

BOOK CONTEXT:
{context}
"""
    payload = _decode_object(_response_text(llm.invoke(prompt)), "questions")
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("The model did not return any quiz questions. Please generate them again.")

    questions = []
    for item in raw_questions[:count]:
        if not isinstance(item, dict):
            continue
        raw_options = item.get("options")
        if isinstance(raw_options, list) and len(raw_options) == len(LETTERS):
            options = dict(zip(LETTERS, raw_options))
        elif isinstance(raw_options, dict):
            options = {letter: raw_options.get(letter) for letter in LETTERS}
        else:
            continue
        if any(not isinstance(options[letter], str) or not options[letter].strip() for letter in LETTERS):
            continue

        answer = item.get("answer")
        if not isinstance(answer, str) or answer.strip().upper() not in LETTERS:
            continue
        try:
            questions.append(
                {
                    "question": _clean_text(item.get("question"), "question", 600),
                    "options": {letter: options[letter].strip()[:400] for letter in LETTERS},
                    "answer": answer.strip().upper(),
                    "explanation": _clean_text(item.get("explanation"), "explanation", 1200),
                    "pages": _valid_pages(item, source_pages),
                }
            )
        except ValueError:
            continue
    if not questions:
        raise ValueError("The model returned no usable quiz questions. Please generate them again.")

    return {
        "id": uuid4().hex,
        "questions": questions,
        "answers": None,
        "score": None,
    }
