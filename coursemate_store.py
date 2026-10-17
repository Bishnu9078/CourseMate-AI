# Copyright (c) 2026 Bishnuprasad Sahoo (GitHub: Bishnu9078)
# Licensed under the MIT License. See LICENSE in the project root.

"""Local SQLite persistence for books, chats, notes, and study progress."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / ".coursemate"
DATABASE_PATH = DATA_DIRECTORY / "coursemate.sqlite3"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class CourseMateStore:
    """Persist CourseMate data on this machine in an ignored local directory."""

    def __init__(self, database_path: str | Path = DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS books (
                    fingerprint TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    pages INTEGER NOT NULL,
                    chunks INTEGER NOT NULL,
                    pdf_data BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    fingerprint TEXT,
                    messages_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chats_updated_at_idx
                    ON chats(updated_at DESC);

                CREATE TABLE IF NOT EXISTS notes (
                    fingerprint TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS study_materials (
                    fingerprint TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (fingerprint, kind)
                );

                CREATE TABLE IF NOT EXISTS chat_study_materials (
                    chat_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (chat_id, fingerprint, kind)
                );
                """
            )
            # Older versions stored study materials only by book. Attach each
            # legacy item to the latest matching conversation with messages;
            # if there is none, use the oldest matching chat. This avoids
            # attaching old progress to a newer blank chat.
            connection.execute(
                """
                INSERT OR IGNORE INTO chat_study_materials
                    (chat_id, fingerprint, kind, payload_json, updated_at)
                SELECT COALESCE(
                    (
                        SELECT chats.chat_id
                        FROM chats
                        WHERE chats.fingerprint = study_materials.fingerprint
                          AND chats.messages_json <> '[]'
                        ORDER BY chats.updated_at DESC
                        LIMIT 1
                    ),
                    (
                        SELECT chats.chat_id
                        FROM chats
                        WHERE chats.fingerprint = study_materials.fingerprint
                        ORDER BY chats.rowid ASC
                        LIMIT 1
                    )
                ), study_materials.fingerprint, study_materials.kind,
                   study_materials.payload_json, study_materials.updated_at
                FROM study_materials
                WHERE EXISTS (
                    SELECT 1
                    FROM chats
                    WHERE chats.fingerprint = study_materials.fingerprint
                )
                """
            )

    def list_books(self) -> list[dict[str, str | int]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT fingerprint, filename, pages, chunks FROM books ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def save_book(self, book: dict[str, str | int], pdf_data: bytes) -> None:
        if not pdf_data:
            raise ValueError("Cannot save an empty PDF.")
        if hashlib.sha256(pdf_data).hexdigest() != str(book["fingerprint"]):
            raise ValueError("The PDF does not match its saved fingerprint.")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO books (fingerprint, filename, pages, chunks, pdf_data, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    filename = excluded.filename,
                    pages = excluded.pages,
                    chunks = excluded.chunks,
                    pdf_data = excluded.pdf_data,
                    updated_at = excluded.updated_at
                """,
                (
                    str(book["fingerprint"]),
                    str(book["filename"]),
                    int(book["pages"]),
                    int(book["chunks"]),
                    sqlite3.Binary(pdf_data),
                    _timestamp(),
                ),
            )

    def get_book_pdf(self, fingerprint: str) -> bytes | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT pdf_data FROM books WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
        if not row:
            return None
        pdf_data = bytes(row["pdf_data"])
        if hashlib.sha256(pdf_data).hexdigest() != fingerprint:
            raise ValueError("The saved PDF failed its integrity check. Re-upload the book to restore it.")
        return pdf_data

    def save_chat(self, chat: dict) -> None:
        messages_json = json.dumps(chat.get("messages", []), ensure_ascii=False)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO chats (chat_id, title, fingerprint, messages_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    title = excluded.title,
                    fingerprint = excluded.fingerprint,
                    messages_json = excluded.messages_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(chat["id"]),
                    str(chat.get("title", "New chat")),
                    chat.get("fingerprint"),
                    messages_json,
                    _timestamp(),
                ),
            )

    def load_chats(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT chat_id, title, fingerprint, messages_json
                FROM chats ORDER BY updated_at DESC
                """
            ).fetchall()

        chats = []
        for row in rows:
            try:
                messages = json.loads(row["messages_json"])
            except (TypeError, json.JSONDecodeError):
                messages = []
            if not isinstance(messages, list):
                messages = []
            chats.append(
                {
                    "id": str(row["chat_id"]),
                    "title": str(row["title"]),
                    "fingerprint": row["fingerprint"],
                    "messages": messages,
                }
            )
        return chats

    def save_notes(self, fingerprint: str, content: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO notes (fingerprint, content, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    content = excluded.content,
                    updated_at = excluded.updated_at
                """,
                (fingerprint, content, _timestamp()),
            )

    def load_notes(self, fingerprint: str) -> str:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT content FROM notes WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
        return str(row["content"]) if row else ""

    def save_study_material(self, fingerprint: str, kind: str, payload: dict) -> None:
        if kind not in {"flashcards", "quiz"}:
            raise ValueError("Unsupported study material type.")
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO study_materials (fingerprint, kind, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(fingerprint, kind) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (fingerprint, kind, payload_json, _timestamp()),
            )

    def load_study_materials(self, fingerprint: str) -> dict[str, dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT kind, payload_json FROM study_materials
                WHERE fingerprint = ?
                """,
                (fingerprint,),
            ).fetchall()

        materials: dict[str, dict] = {}
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                materials[str(row["kind"])] = payload
        return materials

    def save_chat_study_material(
        self, chat_id: str, fingerprint: str, kind: str, payload: dict
    ) -> None:
        if kind not in {"flashcards", "quiz"}:
            raise ValueError("Unsupported study material type.")
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_study_materials
                    (chat_id, fingerprint, kind, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, fingerprint, kind) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (chat_id, fingerprint, kind, payload_json, _timestamp()),
            )

    def load_chat_study_materials(
        self, chat_id: str, fingerprint: str
    ) -> dict[str, dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT kind, payload_json FROM chat_study_materials
                WHERE chat_id = ? AND fingerprint = ?
                """,
                (chat_id, fingerprint),
            ).fetchall()

        materials: dict[str, dict] = {}
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                materials[str(row["kind"])] = payload
        return materials
