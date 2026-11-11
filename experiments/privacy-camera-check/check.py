from concurrent.futures import Future
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
import time
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import uvicorn

import main
import store
import vision

mode = "bright"
hour = 9
samples = 0
entered_cosmos = Event()
finish_cosmos = Event()
finished_cosmos = Event()

class FixtureCapture:
    bright = np.full((720, 1280, 3), 100, np.uint8)
    black = np.zeros((720, 1280, 3), np.uint8)

    def __init__(self, *args):
        pass

    def set(self, *args):
        return True

    def read(self):
        if mode == "missing":
            return False, None
        return True, self.bright if mode == "bright" else self.black

    def release(self):
        pass

class FixtureVision:
    def sample(self, frame, config):
        global samples
        assert mode == "bright"
        samples += 1
        return "low", "sit", None, 1, 2, "unknown", None, "keypoints"

def fixture_cosmos(*args, **kwargs):
    entered_cosmos.set()
    assert finish_cosmos.wait(10)
    finished_cosmos.set()
    return "sitting", "stale image"

def get(path):
    with urlopen("http://127.0.0.1:8765" + path, timeout=2) as reply:
        return reply.read()

def state():
    return json.loads(get("/api/state"))

def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except OSError:
            pass
        time.sleep(.12)
    raise AssertionError("State did not change: " + str(state().get("camera")))

def preview_blocked(private=False):
    try:
        get("/api/calibration/frame")
    except HTTPError as error:
        assert error.code == 503
        if private:
            assert "隐私保护" in json.load(error)["detail"]
    else:
        raise AssertionError("Preview was still visible")

cv2.VideoCapture = FixtureCapture
vision.Vision = FixtureVision
main.datetime = SimpleNamespace(now=lambda _: SimpleNamespace(hour=hour))
main.describe_desk_frames = fixture_cosmos
with TemporaryDirectory(dir="/dev/shm", prefix="orin-camera-privacy-") as folder:
    store.ROOT = Path(folder)
    main.CAL = store.ROOT / "calibration.json"
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=8765, log_level="warning"))
    worker = Thread(target=server.run, daemon=True)
    worker.start()
    try:
        wait_for(lambda: state()["camera"].get("person") == "sit")
        assert get("/api/calibration/frame").startswith(b"\xff\xd8")
        main.events.put_nowait(([b"synthetic"], None, main.vision_epoch))
        assert entered_cosmos.wait(2)
        pending = Future()
        main.events.put_nowait(([b"queued"], pending, main.vision_epoch))
        mode = "black"
        wait_for(lambda: state()["camera"].get("vision_paused"))
        assert not state()["camera"]["privacy_active"]
        preview_blocked()
        assert pending.cancelled()
        samples_before = samples
        wait_for(lambda: state()["camera"]["privacy_active"])
        assert state()["camera"]["ready"]
        preview_blocked(private=True)
        finish_cosmos.set()
        assert finished_cosmos.wait(2)
        wait_for(lambda: state()["totals"].get("privacy", 0) >= 1)
        snapshot = state()
        assert snapshot["camera"]["person"] == snapshot["camera"]["presence"] == "unknown"
        assert not snapshot["notes"] and samples == samples_before
        assert any(i["state"] == "privacy" for i in snapshot["intervals"])
        print("DAY_BLACK: immediate pause, privacy recorded, no preview or stale Cosmos summary", flush=True)
        mode = "bright"
        time.sleep(.8)
        assert state()["camera"]["privacy_active"]
        preview_blocked(private=True)
        wait_for(lambda: not state()["camera"]["vision_paused"])
        assert get("/api/calibration/frame").startswith(b"\xff\xd8")
        print("BRIGHT_RECOVERY: preview and inference resumed after stable light", flush=True)
        hour = 18
        mode = "black"
        wait_for(lambda: state()["camera"]["vision_paused"])
        time.sleep(3.2)
        assert not state()["camera"]["privacy_active"]
        preview_blocked()
        print("NIGHT_BLACK: dark pause, no privacy interval", flush=True)
        hour = 9
        mode = "bright"
        wait_for(lambda: not state()["camera"]["vision_paused"])
        mode = "missing"
        wait_for(lambda: not state()["camera"]["ready"])
        wait_for(lambda: state()["camera"]["privacy_active"])
        assert not state()["camera"]["ready"]
        preview_blocked(private=True)
        print("DAY_MISSING: unavailable camera remains marked unavailable, privacy active", flush=True)
    finally:
        finish_cosmos.set()
        server.should_exit = True
        worker.join(5)
