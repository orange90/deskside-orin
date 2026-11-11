"""Exercise the deployed app routes inside an isolated network namespace."""
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread
import time
from urllib.request import Request, build_opener, ProxyHandler
from uuid import uuid4

import uvicorn

import main
import store

BASE = "http://127.0.0.1:8765"
LOCAL = build_opener(ProxyHandler({}))


def request(path, data=None):
    if data is not None:
        req = Request(BASE + path, data=json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    else:
        req = Request(BASE + path)
    with LOCAL.open(req, timeout=25) as reply:
        return reply.read()


def wait_for(check, seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        try:
            if check():
                return
        except OSError:
            pass
        time.sleep(.15)
    raise TimeoutError("Expected local application state was not observed")


def query(text, request_id=None):
    return json.loads(request("/api/query", {"text": text, "id": str(request_id or uuid4())}))


def main_test():
    interfaces = subprocess.check_output(["ip", "-br", "link"], text=True).splitlines()
    assert [line.split()[0] for line in interfaces] == ["lo"]
    assert not subprocess.check_output(["ip", "-4", "route", "show", "default"]).strip()
    assert not subprocess.check_output(["ip", "-6", "route", "show", "default"]).strip()
    # The real camera stays with the active app. Test the same HTTP routes against a temporary database.
    main.camera = lambda: None
    with TemporaryDirectory(dir="/dev/shm", prefix="orin-offline-dashboard-") as path:
        store.ROOT = Path(path)
        server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=8765, log_level="warning"))
        worker = Thread(target=server.run, daemon=True)
        worker.start()
        try:
            wait_for(lambda: json.loads(request("/api/state")), 8)
            assert 'id="voice"' in request("/").decode()
            now = time.time()
            store.persist("person", "sit", now - 120, False)
            store.persist("person", "sit", now, True)
            assert "2 分钟" in query("我今天坐了多久？")["answer"]
            assert any(word in query("用一句话说明为什么天空是蓝色的。")["answer"]
                       for word in ("散射", "波长"))
            reminder = uuid4()
            assert "已设置" in query("五分钟后提醒我喝水。", reminder)["answer"]
            with store.database() as db:
                db.execute("UPDATE tasks SET due=? WHERE id=?", (time.time() - 1, str(reminder)))
            check = lambda: sum(note["kind"] == "reminder" and note["body"] == "喝水"
                                for note in json.loads(request("/api/state"))["notes"])
            wait_for(lambda: check() == 1, 4)
            time.sleep(1)
            assert check() == 1
            print("Loopback only: deployed dashboard routes, Qwen reply, statistics, and one-time reminder PASS.")
        finally:
            server.should_exit = True
            worker.join(5)


if __name__ == "__main__":
    main_test()
