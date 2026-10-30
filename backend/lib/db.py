"""Yerel kayıt deposu — tarama geçmişi SQLite dosyasında tutulur (MongoDB kurulumu gerekmez).

Veritabanı dosyası: backend/data/elektrovision.db (yoksa otomatik oluşturulur).
"""

import asyncio
import json
import os
import sqlite3
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DB_PATH = Path(os.environ.get("DB_PATH", Path(__file__).parent.parent / "data" / "elektrovision.db"))
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, doc TEXT NOT NULL)"
        )
        _conn.execute("CREATE INDEX IF NOT EXISTS created_desc ON predictions (created_at DESC)")
        _conn.commit()
    return _conn


def _insert(doc: dict) -> None:
    with _lock:
        c = _connection()
        c.execute(
            "INSERT INTO predictions (id, created_at, doc) VALUES (?, ?, ?)",
            (doc["id"], doc["created_at"], json.dumps(doc, ensure_ascii=False)),
        )
        c.commit()


def _latest(limit: int) -> list[dict]:
    with _lock:
        rows = _connection().execute(
            "SELECT doc FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def _delete(id: str) -> bool:
    with _lock:
        c = _connection()
        cur = c.execute("DELETE FROM predictions WHERE id = ?", (id,))
        c.commit()
    return cur.rowcount > 0


def _set_feedback(id: str, label: str) -> None:
    with _lock:
        c = _connection()
        row = c.execute("SELECT doc FROM predictions WHERE id = ?", (id,)).fetchone()
        if row:
            doc = json.loads(row[0])
            doc["feedback_label"] = label
            c.execute("UPDATE predictions SET doc = ? WHERE id = ?", (json.dumps(doc, ensure_ascii=False), id))
            c.commit()


async def set_feedback(id: str, label: str) -> None:
    await asyncio.to_thread(_set_feedback, id, label)


async def save_prediction(doc: dict) -> None:
    """`doc` JSON'a çevrilebilir olmalı (Pydantic: model_dump(mode="json"))."""
    await asyncio.to_thread(_insert, doc)


async def latest_predictions(limit: int = 24) -> list[dict]:
    return await asyncio.to_thread(_latest, limit)


async def delete_prediction(id: str) -> bool:
    return await asyncio.to_thread(_delete, id)


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
