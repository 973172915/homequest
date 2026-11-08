"""界面冒烟测试：三个视图都能正常渲染，对话输入框在顾问视图里存在。"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / "app" / "ui" / "streamlit_app.py"


def test_ui_renders_all_views():
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()

    assert not at.exception, f"页面抛异常：{[str(e.value) for e in at.exception]}"
    assert [r.label for r in at.radio] == ["功能"] or at.radio, "缺少视图切换"
    options = at.radio[0].options
    assert options == ["💬 AI 顾问", "🔍 找房", "💰 估价"]

    # 默认在顾问视图：应该有对话输入框
    assert len(at.chat_input) == 1, "顾问视图缺 chat_input"

    # 切到找房
    at.radio[0].set_value("🔍 找房").run()
    assert not at.exception
    assert any(b.label == "搜索" for b in at.button)

    # 切到估价
    at.radio[0].set_value("💰 估价").run()
    assert not at.exception
    assert any(b.label == "估算价格" for b in at.button)
