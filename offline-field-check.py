"""User-operated end-to-end field acceptance after disconnecting the Orin's uplinks."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8765"


def request(path, payload=None):
    args = {} if payload is None else {
        "data": json.dumps(payload).encode(),
        "headers": {"Content-Type": "application/json"},
    }
    with urlopen(Request(BASE + path, **args), timeout=70) as response:
        return json.load(response)


def assert_isolated():
    # The operator physically disconnects the uplinks, keeping the Orin's local terminal.
    routes = [subprocess.check_output(["ip", "-j", "-4" if v == 4 else "-6", "route", "show", "table", "all"],
                                      text=True) for v in (4, 6)]
    if any(route.get("dst") == "default" and route.get("type", "unicast") == "unicast"
           for text in routes for route in json.loads(text)):
        raise RuntimeError("仍检测到可用默认路由，请在本机关闭 Wi-Fi 并拔掉外网网线后再试。")


def wait_for(get, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        current = request("/api/state")
        if get(current):
            return current
        time.sleep(.4)
    raise RuntimeError("在时限内未达到预期状态，结果需要复测。")


def main():
    parser = argparse.ArgumentParser(description="Orin 现场断外网闭环验收")
    parser.add_argument("--skip-voice", action="store_true", help="麦克风未接时先验视觉和看板")
    args = parser.parse_args()
    assert_isolated()
    for port in (8081, 8082):
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as result:
            assert result.status == 200
    original = request("/api/state")["settings"]
    steps = {"no_default_routes_at_start": True, "voice_test_requested": not args.skip_voice}
    try:
        input("先站在正常使用位置。站稳后按回车。")
        wait_for(lambda state: state["camera"].get("person") == "stand", 8)
        request("/api/settings", {**original, "sit_minutes": 1})
        baseline = time.time()
        input("现在坐下。坐稳后按回车，继续坐到看板出现久坐提醒。")
        wait_for(lambda state: state["camera"].get("person") == "sit", 8)
        steps["sit_observed"] = True
        wait_for(lambda state: any(note["kind"] == "reminder" and note["at"] >= baseline
                                   and "起身活动" in note["body"] for note in state["notes"]), 85)
        steps["reminder_visible"] = input("在看板看到久坐提醒了吗？输入 y 确认。").strip().lower() == "y"
        stood_at = time.time()
        input("现在起身。站稳后按回车。")
        wait_for(lambda state: state["camera"].get("person") == "stand", 8)
        steps["stand_observed"] = True
        wait_for(lambda state: any(note["kind"] == "cosmos" and note["at"] >= stood_at
                                   for note in state["notes"]), 30)
        steps["cosmos_summary_present"] = True
        if not args.skip_voice:
            input("按回车开始实体麦克风测试。")
            request("/api/voice/start", {})
            input("请说「我今天坐了多久」，说完立即按回车。")
            reply = request("/api/voice/stop", {"id": str(uuid4()), "save": False})
            steps["voice_answer_ready_seconds"] = reply.get("answer_ready_seconds")
            print(f"识别到：{reply['query']}\n本机回复：{reply['answer']}")
            steps["voice_answer_correct"] = input("文字回复的坐姿时长正确吗？输入 y 确认。").strip().lower() == "y"
        steps["dashboard_verified"] = input("打开本机 127.0.0.1:8765，核对坐站、提醒和摘要。均可见则输入 y。").strip().lower() == "y"
        assert_isolated()
        steps["no_default_routes_at_start_and_end"] = True
    finally:
        try:
            request("/api/settings", original)
            steps["original_settings_restored"] = True
        finally:
            path = ROOT / "logs" / f"field-offline-{time.strftime('%Y%m%d-%H%M%S')}.json"
            path.write_text(json.dumps(steps, ensure_ascii=False, indent=2) + "\n")
            print(f"离线验收结果保存在本机 {path}。")


if __name__ == "__main__":
    main()
