import json
from pathlib import Path
import re
from runpy import run_path
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread
import time
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen
from uuid import uuid4

import uvicorn
import main
import store
from measure import percentile

suite = Path("runtime/voice-suite-regression")
cases = json.loads((suite / "cases.json").read_text())
helpers = run_path("offline-dashboard-smoke.py")
request, wait_for = helpers["request"], helpers["wait_for"]

def check_case(case):
    request_id = str(uuid4())
    assert json.loads(request("/api/voice/start", {}))["recording"]
    time.sleep(.5)
    subprocess.run(["paplay", "--device", "orin_fixture", str(suite / "fixtures" / (case["id"] + ".wav"))],
                   check=True, timeout=12)
    time.sleep(.4)
    start = time.monotonic()
    row = {"case": case["id"], "kind": case["kind"], "synthetic_noise": case.get("noise", False)}
    try:
        reply = json.loads(request("/api/voice/stop", {"id": request_id, "save": False}))
    except HTTPError as error:
        row.update(status=error.code, error=json.load(error))
    else:
        row.update(status=200, query=reply["query"], answer=reply["answer"],
                   server_seconds=reply["answer_ready_seconds"])
    row["http_seconds"] = round(time.monotonic() - start, 3)
    assert not json.loads(request("/api/state"))["recording"]
    with store.database() as db:
        task = db.execute("SELECT kind,body,due FROM tasks WHERE id=?", (request_id,)).fetchone()
    kind = case["kind"]
    task_ok = task is None
    if kind in {"reminder", "todo"}:
        task_ok = bool(task and task["kind"] == kind and task["body"] == case["want"])
        if task_ok and kind == "reminder":
            task_ok = 5390 < task["due"] - time.time() < 5410
    row["passed"] = bool(re.search(case["want"], row.get("answer", ""))) and task_ok
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row

def resident_transcribe(path, model):
    boundary = "orin-whisper-fixture-boundary"
    headers = (f"--{boundary}\r\n"
               'Content-Disposition: form-data; name="file"; filename="fixture.wav"\r\n'
               "Content-Type: audio/wav\r\n\r\n").encode()
    body = headers + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = Request("http://127.0.0.1:8083/inference", data=body,
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request, timeout=25) as reply:
        transcript = json.load(reply)["text"].strip()
    return transcript

main.camera = lambda: None
main.voice.transcribe = resident_transcribe
with TemporaryDirectory(dir=suite, prefix="fixture-db-") as folder:
    store.ROOT = Path(folder)
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=8765, log_level="warning"))
    worker = Thread(target=server.run, daemon=True)
    worker.start()
    try:
        def model_ready():
            with urlopen("http://127.0.0.1:8081/health", timeout=2) as response:
                return response.status == 200
        wait_for(model_ready, 60)
        wait_for(lambda: request("/api/state"), 8)
        now = time.time()
        store.persist("person", "sit", now - 131, False)
        store.persist("person", "sit", now - 11, True)
        store.persist("person", "stand", now - 11, True)
        store.persist("person", "stand", now, True)
        rows = [check_case(case) for case in cases]
        assert not store.snapshot()["notes"]
        summary = {"samples": len(rows), "passed": sum(row["passed"] for row in rows),
                   "synthetic_noise_samples": sum(row["synthetic_noise"] for row in rows)}
        for category in ("common", "general"):
            group = [row for row in rows if (row["kind"] == "general") == (category == "general")]
            timed = [row["http_seconds"] for row in group if row["status"] == 200]
            summary[category] = {"samples": len(group), "timed": len(timed), "http_p95": percentile(timed, .95)}
        print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    finally:
        server.should_exit = True
        worker.join(5)
