"""用户与登录令牌。

安全要点（这几条是账号系统的底线）：
1. 密码只存 PBKDF2-HMAC-SHA256 哈希 + 每用户独立盐，绝不存明文
2. 校验用 hmac.compare_digest 做常数时间比较，防时序侧信道
3. 登录令牌只存 SHA-256 摘要，数据库泄露也拿不到可用令牌
4. 令牌有有效期，可主动吊销
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta

from housing.storage.sessions import _connect, _now, init

ITERATIONS = 200_000
TOKEN_TTL_DAYS = 7


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS
    ).hex()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def init_auth() -> None:
    init()
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def register(username: str, password: str) -> tuple[int | None, str]:
    """注册。返回 (user_id, 错误信息)。"""
    init_auth()
    username = (username or "").strip()

    if len(username) < 2 or len(username) > 20:
        return None, "用户名需 2~20 个字符"
    if len(password) < 6:
        return None, "密码至少 6 位"

    salt = secrets.token_hex(16)
    with _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, _hash_password(password, salt), salt, _now()),
            )
            return int(cur.lastrowid), ""
        except sqlite3.IntegrityError:
            return None, "用户名已存在"


def verify(username: str, password: str) -> int | None:
    """校验用户名密码，成功返回 user_id。"""
    init_auth()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, password_hash, salt FROM users WHERE username = ?",
            ((username or "").strip(),),
        ).fetchone()
    if row is None:
        # 用户不存在时也跑一次哈希，避免响应时间暴露用户是否存在
        _hash_password(password or "", "00" * 16)
        return None
    if hmac.compare_digest(row["password_hash"], _hash_password(password or "", row["salt"])):
        return int(row["id"])
    return None


def issue_token(user_id: int) -> str:
    """签发登录令牌（明文只返回一次，库里只存摘要）。"""
    init_auth()
    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(days=TOKEN_TTL_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token_hash, user_id, expires_at, created_at) "
            "VALUES (?, ?, ?, ?)",
            (_hash_token(token), user_id, expires, _now()),
        )
    return token


def resolve_token(token: str) -> int | None:
    """令牌换 user_id；过期或不存在返回 None。"""
    if not token:
        return None
    init_auth()
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at FROM auth_tokens WHERE token_hash = ?",
            (_hash_token(token),),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] < _now():
            conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_hash_token(token),))
            return None
    return int(row["user_id"])


def revoke_token(token: str) -> None:
    init_auth()
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_hash_token(token),))


def get_user(user_id: int) -> dict | None:
    init_auth()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None
