"""界面冒烟测试：未登录时应看到登录页，且不抛异常。"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / "app" / "ui" / "streamlit_app.py"


def test_shows_login_when_anonymous():
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()

    assert not at.exception, f"页面抛异常：{[str(e.value) for e in at.exception]}"
    # 未登录：有登录/注册切换和用户名输入框，没有对话框
    assert any(r.label == "模式" for r in at.radio) or at.radio
    assert len(at.text_input) >= 2
    assert len(at.chat_input) == 0, "未登录不应出现对话输入框"


def test_login_form_rejects_empty_submit():
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    # 直接点提交（用户名密码为空）→ 应停留在登录页，不崩溃
    submit = at.form_submit_button[0]
    submit.set_value(True).run()
    assert not at.exception
    assert len(at.chat_input) == 0
