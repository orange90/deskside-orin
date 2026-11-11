"""Test the real loopback Qwen service and tool wiring using an isolated SQLite database."""
import json
from pathlib import Path
import tempfile
import time

import store
from assistant import respond


def check(case, phrase, predicate):
    start = time.monotonic()
    try:
        answer = respond(phrase, case, save=True)
        success = predicate(answer, store.snapshot())
        detail = {"answer": answer, "passed": bool(success)}
    except (OSError, ValueError, KeyError) as error:
        detail = {"error": str(error), "passed": False}
    print(json.dumps({"case": case, "seconds": round(time.monotonic() - start, 3), **detail},
                     ensure_ascii=False), flush=True)
    return detail["passed"]


def main():
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="orin-tool-smoke-") as temp:
        store.ROOT = Path(temp)
        store.initialize()
        now = time.time()
        store.persist("person", "sit", now - 120, False)
        store.persist("person", "sit", now, True)
        results = [
            check("stats", "我今天坐了多久？", lambda text, _: "2 分钟" in text),
            check("general", "用一句话说明为什么天空是蓝色的。",
                  lambda text, _: any(word in text for word in ("散射", "波长"))),
            check("todo", "添加待办：整理书桌。",
                  lambda _, data: sum(row["body"] == "整理书桌" and row["kind"] == "todo"
                                      for row in data["tasks"]) == 1),
            check("todo", "添加待办、整理书桌。",
                  lambda _, data: sum(row["id"] == "todo" for row in data["tasks"]) == 1),
            check("reminder", "五分钟后提醒我喝水。",
                  lambda _, data: any(row["kind"] == "reminder" and "喝水" in row["body"]
                                      and 260 <= row["due"] - time.time() <= 300
                                      for row in data["tasks"])),
        ]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
