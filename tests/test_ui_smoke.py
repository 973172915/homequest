"""界面冒烟测试：用 Streamlit 官方测试框架跑一遍页面脚本，确认能正常渲染。

API 没启动时页面会显示错误提示，但脚本本身不应抛异常。
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / "app" / "ui" / "streamlit_app.py"


def test_streamlit_app_renders_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()

    assert not at.exception, f"页面抛异常：{[str(e.value) for e in at.exception]}"
    assert [t.label for t in at.tabs] == ["🔍 找房", "💬 AI 顾问"]
    assert any(b.label == "搜索" for b in at.button)
    assert any(b.label == "咨询顾问" for b in at.button)
