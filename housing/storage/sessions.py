"""会话存储：SQLite 本地持久化，按用户隔离。

为什么用 SQLite 而不是 JSON 文件：
- 标准库自带，零依赖
- 追加消息是 O(1)，不用每次重写整个文件

表结构：
    sessions(id, user_id, title, created_at, updated_at, summary)
    messages(id, session_id, role, content, created_at)

所有读写都带 user_id 条件，防止改 URL 里的 session_id 就能看别人的会话。
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from housing.config import settings

DB_FILE = "sessions.db"


def _db_path() -> Path:
    settings.artifacts.mkdir(parents=True, exist_ok=True)
    return settings.artifacts / DB_FILE


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_user_column(conn: sqlite3.Connection) -> None:
    """老库没有 user_id 列，补上（默认 0 表示「历史遗留、无归属」）。"""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)")}
    if "user_id" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0")


def init() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL DEFAULT 0,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                summary    TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            """
        )
        _ensure_user_column(conn)


def create(title: str = "新对话", user_id: int = 0) -> str:
    """新建会话，返回 id。"""
    init()
    session_id = uuid.uuid4().hex[:12]
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, title, now, now),
        )
    return session_id


def list_all(user_id: int, limit: int = 50) -> list[dict]:
    """按最近更新排序，只返回该用户的会话。"""
    init()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT s.id, s.title, s.updated_at, s.summary,
                      (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n
               FROM sessions s
               WHERE s.user_id = ?
               ORDER BY s.updated_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get(session_id: str, user_id: int) -> dict | None:
    """取会话完整内容；不属于该用户时返回 None（越权防护）。"""
    init()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
        ).fetchone()
        if row is None:
            return None
        msgs = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return {
        "id": row["id"],
        "title": row["title"],
        "summary": row["summary"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": [dict(m) for m in msgs],
    }


def _owned(conn: sqlite3.Connection, session_id: str, user_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)
    ).fetchone()
    return row is not None


def add_message(session_id: str, role: str, content: str, user_id: int) -> bool:
    init()
    with _connect() as conn:
        if not _owned(conn, session_id, user_id):
            return False
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
    return True


def set_summary(session_id: str, summary: str, user_id: int) -> bool:
    if not summary:
        return False
    init()
    with _connect() as conn:
        if not _owned(conn, session_id, user_id):
            return False
        conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))
    return True


def rename(session_id: str, title: str, user_id: int) -> bool:
    init()
    with _connect() as conn:
        if not _owned(conn, session_id, user_id):
            return False
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title[:40], session_id))
    return True


def delete(session_id: str, user_id: int) -> bool:
    init()
    with _connect() as conn:
        if not _owned(conn, session_id, user_id):
            return False
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return True


def auto_title(text: str) -> str:
    """用第一句话生成标题。"""
    clean = " ".join(str(text).split())
    return clean[:18] + ("…" if len(clean) > 18 else "")
