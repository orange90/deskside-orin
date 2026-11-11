"""Run operator-guided microphone checks against the deployed local API."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import store
from measure import percentile

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8765"
STATS = (
    "我今天坐了多久？", "今天在桌前坐了多少分钟？", "今天坐姿时长是多少？",
    "告诉我今天坐了多长时间。", "今天站了多久？", "站立目标完成了吗？",
    "今天的站立进度如何？", "我今天站立了多长时间？",
)
ITEMS = ("整理书桌", "检查台灯", "给花浇水", "收拾背包", "清洗杯子", "整理书架", "检查电池", "准备雨伞")
GENERAL = (
    "为什么天空通常是蓝色的？", "摄氏零度对应多少华氏度？", "用一句话解释闰年。",
    "七乘八等于多少？", "月亮为什么会有不同形状？", "三角形有几条边？",
)


def request(path, payload=None):
    args = {} if payload is None else {
        "data": json.dumps(payload).encode(),
        "headers": {"Content-Type": "application/json"},
    }
    with urlopen(Request(BASE + path, **args), timeout=70) as response:
        return json.load(response)


def cases():
    for group, phrases in (
        ("stats", STATS),
        ("reminder", tuple(f"九十分钟后提醒我{item}。" for item in ITEMS)),
        ("todo", tuple(f"添加待办：{item}。" for item in ITEMS)),
        ("general", GENERAL),
    ):
        for index, phrase in enumerate(phrases):
            yield group, phrase, index in {len(phrases) - 2, len(phrases) - 1}


def check_tool_row(request_id, group, phrase):
    with store.database() as db:
        row = db.execute("SELECT kind,body,due FROM tasks WHERE id=?", (request_id,)).fetchone()
        if row is not None:
            db.execute("DELETE FROM tasks WHERE id=?", (request_id,))
    if group not in {"todo", "reminder"}:
        return row is None
    prefix = "添加待办：" if group == "todo" else "九十分钟后提醒我"
    expected_body = phrase.removeprefix(prefix).rstrip("。")
    if row is None or row["kind"] != group or row["body"] != expected_body:
        return False
    return group == "todo" or 5280 < row["due"] - time.time() < 5460


def suite_cases(size):
    scenarios = list(cases())
    if size == 30:
        return scenarios
    selected = (1, 5, 8, 13, 16, 17, 24, 25, 28, 30)
    return [scenarios[index - 1] for index in selected]


def main():
    parser = argparse.ArgumentParser(description="Orin 中文语音现场测试")
    parser.add_argument("--text-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--suite", type=int, choices=(10, 30), help="用例数量，新测试默认 10 条")
    parser.add_argument("--resume", type=Path, help="继续同一测试日志")
    parser.add_argument("--start-at", type=int, choices=range(1, 31), help="从指定条目重做")
    args = parser.parse_args()
    with urlopen("http://127.0.0.1:8081/health", timeout=4) as health:
        assert health.status == 200, "Qwen 尚未就绪"
    mode = "text_only"
    path = args.resume or ROOT / "logs" / f"field-voice-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    rows = {}
    previous_size = None
    if args.resume:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "index" in row:
                if row.get("verification") != mode:
                    parser.error("续测模式与之前不一致")
                previous_size = row.get("suite", 30)
                rows[row["index"]] = row
    size = args.suite or previous_size or 10
    if previous_size and size != previous_size:
        parser.error("续测必须使用原用例数量")
    if args.start_at and args.start_at > size:
        parser.error("起始条目超过用例数量")
    start_at = args.start_at or max(rows, default=0) + 1
    print("请核对麦克风识别和文字回答。")
    with path.open("a" if args.resume else "x", encoding="utf-8") as log:
        for count, (group, phrase, noisy) in enumerate(suite_cases(size), 1):
            if count < start_at:
                continue
            request_id = str(uuid4())
            print(f"\n第 {count}/{size} 条 · {'轻微背景噪声' if noisy else '安静'}\n请说：{phrase}")
            input("准备好后按回车开始。")
            row = {"index": count, "category": group, "background_noise": noisy,
                   "verification": mode, "suite": size}
            try:
                request("/api/voice/start", {})
                input("说完立即按回车结束录音。")
                reply = request("/api/voice/stop", {"id": request_id, "save": False})
                row["answer_ready_seconds"] = reply.get("answer_ready_seconds")
                print(f"识别到：{reply['query']}\n本机回复：{reply['answer']}")
                row["operator_confirmed"] = input("文字回复正确吗？输入 y 确认。").strip().lower() == "y"
            except HTTPError as error:
                row["http_status"] = error.code
                print(f"本机语音请求失败，状态码 {error.code}。")
            finally:
                row["tool_ok"] = check_tool_row(request_id, group, phrase)
            row["passed"] = row.get("operator_confirmed", False) and row["tool_ok"]
            rows[count] = row
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
        summary = {"total": len(rows), "passed": sum(row["passed"] for row in rows.values()),
                   "verification": mode}
        for label, general in (("common", False), ("general", True)):
            selected = [row for row in rows.values() if (row["category"] == "general") == general]
            summary[f"{label}_total"] = len(selected)
            answers = [row["answer_ready_seconds"] for row in selected if row.get("answer_ready_seconds") is not None]
            summary[f"{label}_text_timed"] = len(answers)
            summary[f"{label}_answer_ready_p95_seconds"] = percentile(answers, .95)
        log.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    print(f"语音验收结果保存在本机 {path}。")


if __name__ == "__main__":
    main()
