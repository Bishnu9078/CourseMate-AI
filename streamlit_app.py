"""Streamlit reading room for CourseMate AI."""

from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from create_database import index_uploaded_pdf
from main import (
    NO_ANSWER,
    build_prompt,
    create_retriever,
    get_llm,
    retrieve_documents,
)


st.set_page_config(
    page_title="CourseMate AI · Reading room",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .coursemate-hero {
        position: relative;
        overflow: hidden;
        padding: clamp(1.7rem, 4vw, 3.3rem);
        border: 1px solid rgba(173, 159, 255, .22);
        border-radius: 26px;
        background:
            radial-gradient(circle at 88% 18%, rgba(130, 107, 255, .24), transparent 25%),
            radial-gradient(circle at 75% 110%, rgba(31, 118, 173, .2), transparent 34%),
            linear-gradient(118deg, #15162b 0%, #121b31 52%, #102234 100%);
        box-shadow: 0 22px 60px rgba(0, 0, 0, .25);
        margin: .4rem 0 1.4rem;
    }
    .coursemate-kicker {
        display: flex;
        align-items: center;
        gap: .55rem;
        color: #c1b8ff;
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 1.1rem;
    }
    .coursemate-kicker-dot {
        width: .5rem;
        height: .5rem;
        border-radius: 50%;
        background: #a99aff;
        box-shadow: 0 0 0 5px rgba(169, 154, 255, .14), 0 0 18px rgba(169, 154, 255, .7);
    }
    .coursemate-title {
        color: #f4f3ff;
        font-size: clamp(2.1rem, 4.8vw, 3.5rem);
        font-weight: 760;
        letter-spacing: -.055em;
        line-height: 1.02;
        margin: 0;
        overflow-wrap: anywhere;
    }
    .coursemate-title span {
        color: #b9adff;
    }
    .coursemate-copy {
        max-width: 44rem;
        color: #c1c9dc;
        font-size: 1.04rem;
        line-height: 1.65;
        margin: 1rem 0 0;
    }
    .coursemate-pills {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.3rem;
    }
    .coursemate-pills span {
        padding: .4rem .7rem;
        border: 1px solid rgba(185, 173, 255, .22);
        border-radius: 999px;
        background: rgba(10, 14, 29, .42);
        color: #e0dcff;
        font-size: .79rem;
        font-weight: 600;
    }
    .coursemate-section-label {
        color: #b9c3da;
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


APP_STATE_VERSION = 2
if st.session_state.get("_coursemate_state_version") != APP_STATE_VERSION:
    for state_key in (
        "active_book",
        "messages",
        "notice",
        "books",
        "book_indexes",
        "chat_threads",
        "chat_order",
        "active_chat_id",
    ):
        st.session_state.pop(state_key, None)
    st.session_state["_coursemate_state_version"] = APP_STATE_VERSION

for state_key, default_value in (
    ("active_book", None),
    ("messages", []),
    ("notice", None),
    ("books", []),
    ("book_indexes", {}),
    ("chat_threads", {}),
    ("chat_order", []),
):
    st.session_state.setdefault(state_key, default_value)


def configured_api_key() -> str | None:
    """Resolve the current local .env key, falling back to Streamlit secrets."""
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=True)
    environment_key = os.getenv("GROQ_API_KEY")
    if environment_key and environment_key.strip():
        return environment_key.strip()

    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except StreamlitSecretNotFoundError:
        secret_key = None
    if secret_key:
        return str(secret_key).strip()
    return None


def book_fingerprint(book: dict[str, str | int] | None) -> str | None:
    if not book:
        return None
    fingerprint = str(book.get("fingerprint", ""))
    return fingerprint if re.fullmatch(r"[0-9a-f]{64}", fingerprint) else None


def create_chat(book: dict[str, str | int] | None = None) -> str:
    """Start a blank chat and optionally keep the current book selected."""
    chat_id = uuid4().hex
    fingerprint = book_fingerprint(book)
    st.session_state.chat_threads[chat_id] = {
        "title": "New chat",
        "fingerprint": fingerprint,
        "book": book,
        "messages": [],
    }
    st.session_state.chat_order.insert(0, chat_id)
    st.session_state.active_chat_id = chat_id
    st.session_state.active_book = book
    st.session_state.messages = st.session_state.chat_threads[chat_id]["messages"]
    return chat_id


def open_chat(chat_id: str) -> None:
    """Restore the selected chat and its associated book."""
    chat = st.session_state.chat_threads[chat_id]
    st.session_state.active_chat_id = chat_id
    st.session_state.active_book = chat.get("book")
    st.session_state.messages = chat["messages"]
    if chat_id in st.session_state.chat_order:
        st.session_state.chat_order.remove(chat_id)
    st.session_state.chat_order.insert(0, chat_id)


def touch_active_chat() -> None:
    """Move the active conversation to the top of the recent chat list."""
    chat_id = st.session_state.get("active_chat_id")
    if chat_id and chat_id in st.session_state.chat_threads:
        if chat_id in st.session_state.chat_order:
            st.session_state.chat_order.remove(chat_id)
        st.session_state.chat_order.insert(0, chat_id)


def set_active_chat_title(prompt: str) -> None:
    """Name a new conversation as soon as its first prompt is submitted."""
    chat_id = st.session_state.get("active_chat_id")
    chat = st.session_state.chat_threads.get(chat_id) if chat_id else None
    if chat and chat.get("title") == "New chat" and prompt.strip():
        chat["title"] = prompt.replace("\n", " ").strip()[:36]


def title_chat_from_input() -> None:
    set_active_chat_title(st.session_state.get("question_input", ""))


# Keep the current conversation when an app update adds chat history to an
# already-open browser session.
if not st.session_state.chat_threads and (
    st.session_state.messages or st.session_state.active_book
):
    legacy_chat_id = uuid4().hex
    legacy_book = st.session_state.active_book
    legacy_messages = st.session_state.messages
    first_question = next(
        (message["content"] for message in legacy_messages if message.get("role") == "user"),
        "New chat",
    )
    st.session_state.chat_threads[legacy_chat_id] = {
        "title": str(first_question).replace("\n", " ")[:36],
        "fingerprint": book_fingerprint(legacy_book),
        "book": legacy_book,
        "messages": legacy_messages,
    }
    st.session_state.chat_order.insert(0, legacy_chat_id)
    st.session_state.active_chat_id = legacy_chat_id
elif st.session_state.get("active_chat_id") not in st.session_state.chat_threads:
    st.session_state.active_chat_id = None


def render_sources(sources: list[dict[str, str | int]]) -> None:
    if not sources:
        return
    with st.expander(f"Supporting passages · {len(sources)}"):
        for number, source in enumerate(sources, start=1):
            page = source.get("page")
            page_label = f"Page {page}" if page else f"Passage {number}"
            st.markdown(f"**{page_label}**")
            st.text(source["text"])
            if number < len(sources):
                st.divider()


def render_message(message: dict) -> None:
    avatar = ":material/auto_awesome:" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])


def switch_book(book: dict[str, str | int]) -> None:
    chat_id = st.session_state.get("active_chat_id")
    chat = st.session_state.chat_threads.get(chat_id) if chat_id else None
    if chat is None:
        create_chat(book)
        return

    if chat.get("fingerprint") != book_fingerprint(book) and chat["messages"]:
        create_chat(book)
        return

    chat["fingerprint"] = book_fingerprint(book)
    chat["book"] = book
    st.session_state.messages = chat["messages"]
    st.session_state.active_book = book


notice = st.session_state.pop("notice", None)
if notice:
    st.toast(notice, icon=":material/check_circle:")


with st.sidebar:
    st.markdown("## :material/auto_stories: CourseMate")
    st.caption("Your personal AI reading room")

    if st.button(
        "New chat",
        icon=":material/add_comment:",
        width="stretch",
        key="new_chat_button",
    ):
        create_chat(st.session_state.active_book)
        st.rerun()

    st.markdown("<div class='coursemate-section-label'>Chats</div>", unsafe_allow_html=True)
    if st.session_state.chat_order:
        for chat_id in st.session_state.chat_order:
            chat = st.session_state.chat_threads.get(chat_id)
            if chat is None:
                continue
            if st.button(
                str(chat.get("title", "New chat")),
                icon=":material/chat_bubble_outline:",
                type="primary" if chat_id == st.session_state.active_chat_id else "secondary",
                width="stretch",
                key=f"open_chat_{chat_id}",
            ):
                open_chat(chat_id)
                st.rerun()
    else:
        st.caption("Your conversations will appear here.")

    st.divider()

    active_book = st.session_state.active_book
    st.markdown("<div class='coursemate-section-label'>Now reading</div>", unsafe_allow_html=True)
    if active_book:
        st.text(str(active_book.get("filename", "Your book")))
        st.caption(f"{active_book.get('pages', 0)} pages · {active_book.get('chunks', 0)} searchable passages")
    else:
        st.caption("Upload a PDF to start a study session.")

    book_lookup = {
        str(book["fingerprint"]): book
        for book in st.session_state.books
        if book_fingerprint(book)
    }
    if book_lookup:
        st.divider()
        st.markdown("<div class='coursemate-section-label'>Your bookshelf · this session</div>", unsafe_allow_html=True)
        selected_fingerprint = st.selectbox(
            "Choose an indexed book",
            options=[""] + list(book_lookup),
            format_func=lambda fingerprint: (
                "Select a book" if not fingerprint else str(book_lookup[fingerprint].get("filename", "Untitled book"))
            ),
            label_visibility="collapsed",
            key="bookshelf_selection",
        )
        if st.button(
            "Open book",
            icon=":material/menu_book:",
            width="stretch",
            disabled=not selected_fingerprint,
            key="open_book_button",
        ):
            switch_book(book_lookup[selected_fingerprint])
            st.session_state.notice = f"Opened {book_lookup[selected_fingerprint].get('filename', 'your book')}"
            st.rerun()

    st.divider()
    st.caption("PDF indexes stay in memory for this session. Upload again after restarting the app.")


active_book = st.session_state.active_book
safe_name = escape(str(active_book.get("filename", "your book"))) if active_book else ""
if active_book:
    hero_title = f"Let's get curious about<br/><span>{safe_name}</span>"
    hero_copy = "Ask questions, explore key ideas, and follow each answer back to the pages that support it."
    hero_pills = "<span>Grounded in your book</span><span>MMR page retrieval</span><span>Source excerpts included</span>"
else:
    hero_title = "Turn reading into<br/><span>real understanding.</span>"
    hero_copy = "Bring a PDF book into your reading room. CourseMate finds the right passages and helps you study them in a conversation."
    hero_pills = "<span>Upload a PDF</span><span>Ask in plain language</span><span>See supporting pages</span>"

st.markdown(
    f"""
    <section class="coursemate-hero">
        <div class="coursemate-kicker"><span class="coursemate-kicker-dot"></span>YOUR PERSONAL STUDY COMPANION</div>
        <h1 class="coursemate-title">{hero_title}</h1>
        <p class="coursemate-copy">{hero_copy}</p>
        <div class="coursemate-pills">{hero_pills}</div>
    </section>
    """,
    unsafe_allow_html=True,
)


with st.expander("Add a book or switch to another PDF", expanded=active_book is None):
    st.markdown("### Build your reading room")
    st.caption("Upload a text-based PDF. The first indexing run downloads the local embedding model and can take a few minutes.")
    uploaded_book = st.file_uploader(
        "Drop a PDF book here",
        type=["pdf"],
        max_upload_size=50,
        key="book_upload",
        help="PDF only · up to 50 MB",
    )
    if uploaded_book is not None:
        size_mb = len(uploaded_book.getvalue()) / (1024 * 1024)
        st.caption(f"{uploaded_book.name} · {size_mb:.1f} MB")

    if st.button(
        "Index this book",
        type="primary",
        icon=":material/auto_awesome:",
        width="stretch",
        disabled=uploaded_book is None,
        key="index_book_button",
    ):
        try:
            with st.status("Making your book searchable", expanded=True) as progress:
                progress.write("Checking the PDF and extracting its pages…")
                book_info, vectorstore = index_uploaded_pdf(uploaded_book.getvalue(), uploaded_book.name)
                st.session_state.book_indexes[book_info["fingerprint"]] = vectorstore
                st.session_state.books = [
                    book for book in st.session_state.books
                    if book.get("fingerprint") != book_info["fingerprint"]
                ]
                st.session_state.books.insert(0, book_info)
                progress.write(f"Created {book_info['chunks']} searchable passages.")
                progress.update(label="Your book is ready", state="complete", expanded=False)

            switch_book(book_info)
            st.session_state.notice = f"Ready to study {book_info['filename']}"
            st.rerun()
        except Exception as error:
            st.error(f"This PDF could not be indexed: {error}")


active_book = st.session_state.active_book
if active_book:
    first, second, third = st.columns(3)
    with first:
        st.metric("Pages in this book", active_book.get("pages", 0), border=True)
    with second:
        st.metric("Searchable passages", active_book.get("chunks", 0), border=True)
    with third:
        st.metric("Answer style", "Book-grounded", border=True)
    st.markdown("### Your study session")
    st.caption("CourseMate searches relevant pages before every answer.")
else:
    st.info("Your reading room is ready. Add a PDF above or open one from your bookshelf.", icon=":material/menu_book:")


for message in st.session_state.messages:
    render_message(message)


suggestions = [
    ("Give me a quick summary", ":material/summarize:"),
    ("What are the key ideas?", ":material/lightbulb:"),
    ("Quiz me on this book", ":material/school:"),
]
suggested_prompt = None
if active_book and not st.session_state.messages:
    st.markdown("#### Start with a question")
    suggestion_columns = st.columns(len(suggestions))
    for column, (label, icon) in zip(suggestion_columns, suggestions):
        with column:
            if st.button(
                label,
                icon=icon,
                width="stretch",
                key=f"suggestion_{label}",
                on_click=set_active_chat_title,
                args=(label,),
            ):
                suggested_prompt = label


typed_prompt = st.chat_input(
    "Ask a question about your book…",
    disabled=active_book is None,
    key="question_input",
    on_submit=title_chat_from_input,
)
question = typed_prompt or suggested_prompt

if question and active_book:
    active_chat_id = st.session_state.get("active_chat_id")
    if active_chat_id and active_chat_id in st.session_state.chat_threads:
        active_chat = st.session_state.chat_threads[active_chat_id]
        if active_chat.get("title") == "New chat":
            active_chat["title"] = question.replace("\n", " ").strip()[:36]
        touch_active_chat()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    api_key = configured_api_key()
    answer = NO_ANSWER
    sources: list[dict[str, str | int]] = []

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        if not api_key:
            answer = "Set GROQ_API_KEY in your local .env file or Streamlit secrets, then ask again."
            st.info(answer)
        else:
            fingerprint = book_fingerprint(active_book)
            vectorstore = st.session_state.book_indexes.get(fingerprint) if fingerprint else None
            if vectorstore is None:
                answer = "This book index is unavailable. Re-upload and index the PDF to continue."
                st.error(answer)
            else:
                try:
                    with st.status("Searching your book", expanded=False) as progress:
                        retriever = create_retriever(vectorstore)
                        documents = retrieve_documents(question, retriever)
                        if documents:
                            progress.update(label="Writing a grounded answer", state="running")
                            response = get_llm(api_key).invoke(build_prompt(question, documents))
                            answer = str(response.content)
                            sources = [
                                {
                                    "page": int(document.metadata["page"]) + 1
                                    if str(document.metadata.get("page", "")).isdigit()
                                    else "",
                                    "text": document.page_content,
                                }
                                for document in documents
                            ]
                        else:
                            answer = NO_ANSWER
                        progress.update(label="Answer ready", state="complete", expanded=False)
                    st.markdown(answer)
                    render_sources(sources)
                except Exception:
                    answer = "I couldn't complete that answer. Check the Groq API key and connection, then try again."
                    st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
