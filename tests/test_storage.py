"""存储层测试：账号、令牌、会话归属（越权防护）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from housing.storage import sessions as store  # noqa: E402
from housing.storage import users  # noqa: E402


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """把数据库指向临时文件，避免污染真实 artifacts/sessions.db。"""
    monkeypatch.setattr(store, "_db_path", lambda: tmp_path / "test.db")
    return tmp_path / "test.db"


def test_register_and_verify(tmp_db):
    uid, err = users.register("alice", "secret123")
    assert uid and err == ""
    assert users.verify("alice", "secret123") == uid
    assert users.verify("alice", "wrong-password") is None
    assert users.verify("nobody", "secret123") is None


def test_duplicate_username_rejected(tmp_db):
    users.register("bob", "secret123")
    uid, err = users.register("bob", "another123")
    assert uid is None
    assert "已存在" in err


def test_weak_password_rejected(tmp_db):
    uid, err = users.register("carol", "123")
    assert uid is None
    assert "6 位" in err


def test_password_not_stored_in_plaintext(tmp_db):
    users.register("dave", "secret123")
    with store._connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username='dave'").fetchone()
    assert "secret123" not in row["password_hash"]
    assert len(row["password_hash"]) == 64      # sha256 hex


def test_token_lifecycle(tmp_db):
    uid, _ = users.register("erin", "secret123")
    token = users.issue_token(uid)
    assert users.resolve_token(token) == uid
    users.revoke_token(token)
    assert users.resolve_token(token) is None
    assert users.resolve_token("not-a-real-token") is None


def test_sessions_are_isolated_between_users(tmp_db):
    a, _ = users.register("ua", "secret123")
    b, _ = users.register("ub", "secret123")

    sid = store.create(user_id=a)
    store.add_message(sid, "user", "我是 A 的会话", a)

    # A 能看到
    assert store.get(sid, a)["messages"][0]["content"] == "我是 A 的会话"
    assert any(s["id"] == sid for s in store.list_all(a))

    # B 看不到，也改不了、删不了
    assert store.get(sid, b) is None
    assert all(s["id"] != sid for s in store.list_all(b))
    assert store.add_message(sid, "user", "B 想插话", b) is False
    assert store.delete(sid, b) is False
    assert store.rename(sid, "B 改的标题", b) is False

    # 原来的会话仍然完好
    assert store.get(sid, a)["title"] != "B 改的标题"
