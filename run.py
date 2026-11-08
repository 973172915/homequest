"""一键启动：建索引（如缺失）→ 起 API → 起界面 → 打开浏览器。

用法：
    python run.py                  # 默认 8000/8501
    python run.py --no-browser     # 不自动开浏览器
    python run.py --rebuild        # 强制重建向量索引
    python run.py --api-only       # 只起后端
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "artifacts" / "index"

# 本机跑绑 127.0.0.1 就够；容器里必须绑 0.0.0.0，否则端口映射进不来
HOST = os.environ.get("APP_HOST", "127.0.0.1")


def wait_for(url: str, timeout: float = 60.0, label: str = "") -> bool:
    """轮询直到服务可用。uvicorn 预热要 10 秒左右。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    print(f"  ✗ {label or url} 在 {timeout:.0f}s 内没起来")
    return False


def build_index_if_needed(rebuild: bool) -> None:
    needed = rebuild or not (INDEX_DIR / "faiss_listings").exists()
    if not needed:
        print("✓ 向量索引已存在")
        return
    print("… 构建向量索引（首次约 2~3 分钟，CPU）")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_index.py")] + (
        ["--rebuild"] if rebuild else []), check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="潍房智估一键启动")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    env_file = ROOT / ".env"
    if not env_file.exists():
        print("! 没有 .env，AI 顾问会以「检索 + 模板摘要」降级模式运行")
        print("  需要完整功能请复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY")

    build_index_if_needed(args.rebuild)

    api_url = f"http://127.0.0.1:{args.api_port}"
    ui_url = f"http://127.0.0.1:{args.ui_port}"

    print(f"… 启动 API  {api_url}")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app",
         "--host", HOST, "--port", str(args.api_port)],
        cwd=ROOT,
    )

    ui: subprocess.Popen | None = None
    try:
        if not wait_for(f"{api_url}/health", 90, "API"):
            print("API 启动失败，退出")
            return 1
        try:
            health = json.loads(urllib.request.urlopen(f"{api_url}/health", timeout=5).read())
            print(f"✓ API 就绪（预热 {health.get('warmup_seconds')}s，"
                  f"LLM {'已配置' if health.get('llm') else '未配置'}）")
        except Exception:
            print("✓ API 就绪")

        if args.api_only:
            print(f"\n后端已启动：{api_url}/docs")
            api.wait()
            return 0

        print(f"… 启动界面 {ui_url}")
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py",
             "--server.headless", "true", "--server.address", HOST,
             "--server.port", str(args.ui_port),
             "--browser.gatherUsageStats", "false"],
            cwd=ROOT,
        )
        if not wait_for(ui_url, 60, "界面"):
            return 1

        print("\n" + "=" * 52)
        print(f"  界面  {ui_url}")
        print(f"  接口  {api_url}/docs")
        print("  Ctrl+C 停止")
        print("=" * 52 + "\n")

        if not args.no_browser:
            webbrowser.open(ui_url)

        # 任一进程退出就收工
        while True:
            if api.poll() is not None:
                print("API 进程退出")
                return api.returncode or 0
            if ui is not None and ui.poll() is not None:
                print("界面进程退出")
                return ui.returncode or 0
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n正在停止…")
        return 0
    finally:
        for proc in (ui, api):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
