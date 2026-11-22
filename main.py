import base64
import json
import logging
import os
import queue
import signal
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

import store
import model_registry
from dashboard_data import dashboard_snapshot
from telemetry import monitor
from assistant import Voice, describe_desk_frames, respond
from state import PersonTrackEvidence, OccupiedPosture, AbsenceEvidence, FrameLight, PresenceContinuity, SittingSession, StableState

LOG = logging.getLogger("uvicorn.error")
ROOT = store.ROOT
CAL = ROOT / "calibration.json"
voice = Voice()
stop = threading.Event()
events = queue.Queue(maxsize=1)
lock = threading.Lock()
latest = {"ready": False}
preview = None
vision_epoch = 0
last_publication = None


def invalidate_scene_locked():
    global vision_epoch
    vision_epoch += 1
    try:
        _, pending, _, _, _ = events.get_nowait()
    except queue.Empty:
        return
    if pending is not None:
        pending.cancel()


def publish(frame, **values):
    global latest, preview, last_publication
    with lock:
        if latest.get("ready") and not latest.get("vision_paused") and (
                not values.get("ready") or values.get("vision_paused")):
            invalidate_scene_locked()
        last_publication = time.monotonic()
        latest = values
        preview = frame


class PostureReview:
    # A reply can have waited behind voice work or another inference.
    def deadline(self, interval, empty_candidate):
        return self.requested + (max(4, interval + 2) if empty_candidate else 7)

    def __init__(self):
        self.pending = None
        self.requested = 0.0
        self.label = None
        self.valid_until = self.label_at = 0.0
        self.absence = AbsenceEvidence()
        self.empty_candidate = None

    def clear(self):
        if self.pending is not None:
            self.pending.cancel()
        self.pending = self.label = None
        self.requested = self.valid_until = self.label_at = 0.0
        self.absence = AbsenceEvidence()
        self.empty_candidate = None

    def invalidate_before(self, first_frame):
        # A cached reply and the next pending reply may come from different
        # frames. Preserve a pending request that already saw the new pose.
        if self.label_at < first_frame:
            self.label = None
        if self.pending is not None and self.requested < first_frame:
            self.pending.cancel()
            self.pending = None
            self.requested = 0.

    def update(self, frame, now, *, empty_candidate, interval):
        if self.empty_candidate is not None and self.empty_candidate != empty_candidate:
            self.clear()
        self.empty_candidate = empty_candidate
        if self.pending is not None and self.pending.done():
            label = None if self.pending.cancelled() else self.pending.result()
            if now >= self.deadline(interval, empty_candidate):
                label = None
            confirmed_empty = self.absence.update(label, self.requested, empty_candidate)
            self.label = None
            if label in {"sitting", "standing", "present", "occluded"} or confirmed_empty:
                self.label = {"sitting": "sit", "standing": "stand", "present": "present",
                              "occluded": "occluded", "away": "away"}[label]
                self.valid_until = self.deadline(interval, empty_candidate)
                self.label_at = self.requested
            LOG.info("Presence review label=%s empty_candidate=%s confirmed_empty=%s", label, empty_candidate, confirmed_empty)
            self.pending = None
        if not empty_candidate:
            self.absence.update(None, now, False)
            if self.label == "away":
                self.label = None
        if self.pending is None and now - self.requested >= interval:
            reply = Future()
            try:
                deadline = now + (max(4, interval + 2) if empty_candidate else 7)
                events.put_nowait(([frame()], reply, vision_epoch,
                                   "occupancy" if empty_candidate else "person", deadline))
            except queue.Full:
                pass
            else:
                self.pending, self.requested = reply, now
        return self.label if now < self.valid_until else None


def desk_image(frame, roi, *, posture_detail=False):
    import cv2
    from vision import crop

    if posture_detail and max(frame.shape[:2]) <= 640:
        # Preserve local pixels and chair context on low-resolution feeds.
        # Use the configured review area for the usual high-resolution feed.
        roi = (0, 0, 1, 1)
    region, _, _ = crop(frame, roi)
    height, width = region.shape[:2]
    scale = min(1, (960 if posture_detail else 640) / max(height, width))
    _, image = cv2.imencode(".jpg", cv2.resize(region, (int(width*scale), int(height*scale))))
    return image.tobytes()


def camera():
    try:
        capture_frames()
    except Exception:
        LOG.exception("Vision loop failed; stopping the app for supervised restart")
        publish(None, ready=False, error="视觉初始化或推理失败，服务正在重启")
        if not stop.is_set():
            os.kill(os.getpid(), signal.SIGTERM)


def capture_frames():
    import cv2
    from vision import Vision, crop

    cv2.setNumThreads(1)
    posture, desk, facing = StableState(), StableState(), StableState()
    session = SittingSession()
    continuity = PresenceContinuity()
    estimate = OccupiedPosture()
    track = PersonTrackEvidence()
    history = deque(maxlen=3)
    cap = None
    last_sample = None
    frame_count = 0
    last_db = 0.0
    last_recorded = None
    previous = "unknown"
    scene_config = None
    multiple = False
    observation_broken = False
    light = FrameLight()
    privacy = False
    last_subject_roi = None
    review = PostureReview()
    try:
        vision = Vision()
        while not stop.is_set():
            start = time.monotonic()
            if cap is None:
                cap = cv2.VideoCapture(int(os.environ.get("ORIN_CAMERA", "0")), cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            ok = bool(ok and frame is not None and frame.size)
            now = time.monotonic()
            captured_at = time.time()
            was_paused, was_private = light.paused, privacy
            brightness = float(frame[::24, ::24].mean()) if ok else None
            paused = light.update(now, brightness)
            privacy = light.sustained and 8 <= datetime.now(store.TZ).hour < 18
            if ok:
                config = json.loads(CAL.read_text()) if CAL.exists() else {}
                person_roi = config.get("person_roi")
                geometry = (frame.shape[:2], person_roi, config.get("cosmos_roi"), config.get("desk_roi"),
                            config.get("low"), config.get("high"))
                gap = last_sample is not None and now - last_sample >= 3
                scene_changed = scene_config is not None and geometry != scene_config
                if gap or scene_changed or (paused and not was_paused):
                    with lock:
                        invalidate_scene_locked()
                    vision.boxes = None
                    review.clear()
                    posture, desk, facing = (StableState() for _ in range(3))
                    history.clear()
                    last_subject_roi = None
                    continuity, estimate, track = PresenceContinuity(), OccupiedPosture(), PersonTrackEvidence()
                    session = SittingSession()
                    previous, last_sample, multiple = "unknown", None, False
                    observation_broken |= gap or scene_changed
                scene_config = geometry
                if paused:
                    ds = ps = present = orientation = "unknown"
                    edge_y = count = region_hint = pose_source = None
                    preview_data = None
                    presence_source = "paused"
                    reviewed = None
                else:
                    d, p, edge_y, count, strong, orientation, region_hint, pose_source = vision.sample(frame, config)
                    multi = count > 1
                    if multi != multiple or vision.overlay.get("subject_changed"):
                        # The current skeleton and a previous in-flight reply
                        # may describe different people.
                        with lock:
                            invalidate_scene_locked()
                        review.clear()
                        history.clear()
                        last_subject_roi = None
                        continuity, estimate, track = PresenceContinuity(), OccupiedPosture(), PersonTrackEvidence()
                        posture, session, facing = StableState(), SittingSession(), StableState()
                        previous = "unknown"
                    multiple = multi
                    tracked = track.observe(count, strong, vision.overlay.get("keypoints", []), now)
                    hint = estimate.joint_hint(vision.overlay) if tracked and not multi else None
                    if estimate.observe_joints(hint, now):
                        review.invalidate_before(estimate.joint_since)
                    at_desk = "present" if count else "unknown"
                    reviewed = None
                    if (p == "unknown" and tracked and count == 1) or (not count and p == "away"):
                        roi = config.get("cosmos_roi") or person_roi or (0, 0, 1, 1)
                        empty_candidate = count == 0
                        interval = (6 if previous == "away" else 3) if empty_candidate else 1.5
                        reviewed = review.update(lambda: desk_image(frame, roi, posture_detail=not empty_candidate), now,
                                                 empty_candidate=empty_candidate, interval=interval)
                        if reviewed in {"sit", "stand", "present"}:
                            p = "unknown" if reviewed == "present" else reviewed
                            at_desk, pose_source = "present", "cosmos"
                        elif reviewed == "away" and empty_candidate:
                            p, at_desk, pose_source = "away", "away", "cosmos"
                        elif empty_candidate:
                            p = at_desk = "unknown"
                    else:
                        review.clear()
                    if at_desk == "present" and not tracked:
                        at_desk, p, pose_source = "unknown", "unknown", None
                    at_desk, presence_source = continuity.update(at_desk, now)
                    # Track confirmation and clear-empty votes already debounce
                    # presence. All consumers use this same answer on this frame.
                    present = at_desk
                    ds = desk.update(d, now, .6)
                    if multi and present == "present":
                        ps, pose_source = "unknown", "multiple_people"
                        orientation, region_hint = "unknown", None
                    elif present != "present":
                        posture, estimate = StableState(), OccupiedPosture()
                        ps, pose_source = present, "cosmos" if present == "away" else None
                    else:
                        stable = posture.update(p, now, 1.2)
                        ps, pose_source = estimate.choose(p, stable, pose_source, True, now,
                                                          (vision.overlay or {}) if tracked else {})
                    _, data = cv2.imencode(".jpg", frame)
                    preview_data = data.tobytes()
                frame_count += 1
                publish(preview_data, ready=True, person=ps, desk=ds, desk_edge=edge_y,
                        pose_source=pose_source, people=count, presence=present,
                        vision_model=vision.model_info, vision_ms=vision.model.last_ms if not paused else None,
                        presence_source=presence_source, occluded=reviewed == "occluded",
                        overlay=vision.overlay if not paused else None,
                        image_width=frame.shape[1], image_height=frame.shape[0], captured_at=captured_at,
                        vision_paused=paused, privacy_active=privacy,
                        orientation=facing.update(orientation, now, .7),
                        frame_count=frame_count, frame_clock=now,
                        fps=round(1 / max(now - last_sample, .001), 1) if last_sample else 0)
                elapsed = now - last_sample if last_sample is not None else 0.
                reminder_pose = "unknown" if (pose_source or "").startswith("estimate_") else ps
                if session.update(reminder_pose, elapsed, sit_limit):
                    store.note("reminder", "已经坐了一段时间，可以起身活动一下。")
                if region_hint:
                    last_subject_roi = region_hint
                roi = config.get("cosmos_roi") or last_subject_roi
                if not paused and not multiple and roi and (not history or now - history[-1][0] >= .75):
                    history.append((now, desk_image(frame, roi)))
                if ps != previous and previous in {"sit", "stand", "away"} and ps in {"sit", "stand", "away"} and history:
                    try:
                        events.put_nowait(([data for _, data in history], None, vision_epoch, None, now + 8))
                    except queue.Full:
                        pass
                previous, last_sample = ps, now
            else:
                publish(None, ready=False, vision_paused=True, privacy_active=privacy, error="摄像头不可用")
                cap.release()
                cap = None
                vision.boxes = None
                review.clear()
                history.clear()
                last_subject_roi = None
                posture, desk, facing = (StableState() for _ in range(3))
                session = SittingSession()
                multiple = False
                ds = ps = present = previous = "unknown"
                last_sample = None
                continuity = PresenceContinuity()
                estimate = OccupiedPosture()
                track = PersonTrackEvidence()
            estimated = ok and not paused and (pose_source or "").startswith("estimate_")
            recorded = "estimated_" + ps if estimated else ps
            states = {"desk": ds, "person": "privacy" if privacy else recorded, "presence": present}
            # Transitions take effect at this frame's acquisition time. The
            # one-second heartbeat only throttles unchanged observations.
            if states != last_recorded or now - last_db >= 1 or observation_broken:
                continuous = bool(not observation_broken and last_db and now - last_db < 3 and (ok or was_paused))
                store.persist_sample(states, captured_at, continuous)
                last_recorded = states
                observation_broken = False
                last_db = now
            if not ok:
                stop.wait(2)
            stop.wait(max(0, .16 - (time.monotonic() - start)))
    finally:
        if cap is not None:
            cap.release()


def cosmos():
    while not stop.is_set():
        try:
            frames, reply, epoch, review_mode, deadline = events.get(timeout=.5)
        except queue.Empty:
            continue
        while epoch == vision_epoch and time.monotonic() < deadline and (voice.busy.is_set() or voice.recording):
            if stop.wait(.15):
                return
        if epoch != vision_epoch or time.monotonic() >= deadline:
            if reply is not None:
                reply.cancel()
            continue
        if reply is not None and not reply.set_running_or_notify_cancel():
            continue
        started = time.monotonic()
        count = int(os.environ.get("ORIN_COSMOS_FRAMES", "1"))
        images = [{"type": "image_url", "image_url": {"url":
                   "data:image/jpeg;base64," + base64.b64encode(frame).decode()}}
                  for frame in frames[-count:]]
        try:
            label, summary = describe_desk_frames(images, posture_review=reply is not None, review_mode=review_mode)
            with lock:
                if reply is not None:
                    reply.set_result(label if epoch == vision_epoch and time.monotonic() < deadline else None)
                    continue
                if epoch != vision_epoch or time.monotonic() >= deadline:
                    continue
                store.note("cosmos", summary)
            LOG.info("Cosmos event took %.3fs, frames=%d, label=%s", time.monotonic() - started, len(images), label)
        except Exception as error:
            if reply is not None:
                reply.set_result(None)
            LOG.warning("Cosmos request failed: %s", type(error).__name__)


sit_limit = 50 * 60


def reminder_loop():
    global sit_limit
    while not stop.wait(.5):
        with store.database() as db:
            sit_limit = db.execute("SELECT value FROM settings WHERE key='sit_minutes'").fetchone()[0] * 60
        store.record_due_reminders()


def capture_watchdog():
    # An unplugged camera still publishes "unavailable". A native capture call
    # that never returns cannot reach the normal reconnect or exception path.
    started = time.monotonic()
    while not stop.wait(2):
        heartbeat = last_publication
        elapsed = time.monotonic() - (started if heartbeat is None else heartbeat)
        if elapsed > (90 if heartbeat is None else 30):
            LOG.error("No capture heartbeat for %.1fs; requesting supervised restart", elapsed)
            os.kill(os.getpid(), signal.SIGTERM)
            return


@asynccontextmanager
async def lifespan(app):
    global last_publication
    last_publication = None
    store.initialize()
    stop.clear()
    workers = [threading.Thread(target=f, daemon=True) for f in (camera, cosmos, reminder_loop, capture_watchdog)]
    workers.append(threading.Thread(target=monitor.run, args=(stop,), daemon=True))
    for worker in workers:
        worker.start()
    yield
    stop.set()
    for worker in workers:
        worker.join(timeout=2)


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])


@app.middleware("http")
async def local_requests(request, call_next):
    origin = request.headers.get("origin")
    if request.method == "POST" and origin and origin not in {
        "http://127.0.0.1:8765", "http://localhost:8765"
    }:
        return Response(status_code=403)
    return await call_next(request)


class Region(BaseModel):
    box: tuple[float, float, float, float]


@app.post("/api/region/{kind}")
def set_region(kind: str, region: Region):
    if kind not in {"desk", "person", "cosmos"}:
        raise HTTPException(404)
    x1, y1, x2, y2 = region.box
    if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1 and x2-x1 >= .05 and y2-y1 >= .05):
        raise HTTPException(422, "选区应至少覆盖画面宽和高的 5%")
    with lock:
        config = json.loads(CAL.read_text()) if CAL.exists() else {}
        if kind == "desk" and config.get("desk_roi") != list(region.box):
            config.pop("low", None)
            config.pop("high", None)
        config[f"{kind}_roi"] = region.box
        save_calibration(config)
    return {"ok": True}


def save_calibration(config):
    tmp = CAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2))
    tmp.replace(CAL)


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    id: uuid.UUID
    save: bool = False


class Record(BaseModel):
    id: uuid.UUID
    save: bool = False


class Settings(BaseModel):
    stand_minutes: int = Field(ge=1, le=1440)
    sit_minutes: int = Field(ge=1, le=1440)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (ROOT / "dashboard.html").read_text()


def camera_view():
    """Published frames are immutable. A stalled capture must not look live."""
    sample = latest
    if sample.get("ready") and time.monotonic() - sample.get("frame_clock", 0) > 3:
        return {"ready": False, "stale": True, "vision_paused": True,
                "person": "unknown", "presence": "unknown", "people": None,
                "privacy_active": bool(sample.get("privacy_active")),
                "error": "摄像头没有新画面"}
    return sample


@app.get("/api/state")
def state():
    return {**store.snapshot(), "camera": camera_view(), "recording": bool(voice.recording)}


class ModelSelection(BaseModel):
    component: str
    model: str


@app.get("/api/reminders")
def reminder_feed(after: int = 0):
    with store.database() as db:
        rows = [dict(row) for row in db.execute(
            "SELECT id,body,at FROM notes WHERE kind='reminder' AND id>? ORDER BY id LIMIT 100",
            (max(0, after),))]
        newest = db.execute("SELECT COALESCE(MAX(id),0) FROM notes WHERE kind='reminder'").fetchone()[0]
    return {"reminders": rows, "latest_id": newest}


@app.get("/api/models")
def model_catalog():
    return model_registry.catalog()


@app.post("/api/models/select")
def model_select(selection: ModelSelection):
    if selection.component != "asr":
        raise HTTPException(400, "此部件暂无其他已适配模型")
    with voice.lock:
        if voice.recording or voice.busy.is_set():
            raise HTTPException(409, "请等待当前语音任务结束再切换")
        try:
            model_registry.select_asr(selection.model)
        except ValueError as error:
            raise HTTPException(400, str(error))
    return {"selected": selection.model, "message": "已切换，下次语音提问生效"}


@app.get("/api/hardware")
def hardware():
    return monitor.snapshot()


@app.get("/api/dashboard")
def dashboard_history(date: str | None = None):
    try:
        return {**dashboard_snapshot(date), "camera": camera_view(), "recording": bool(voice.recording)}
    except ValueError:
        raise HTTPException(400, "日期格式应为 YYYY-MM-DD")


@app.get("/api/live/frame")
def live_frame():
    with lock:
        sample = camera_view()
        if preview is None or not sample.get("ready") or sample.get("vision_paused") or sample.get("privacy_active"):
            raise HTTPException(423, "隐私保护或画面不可用，实时预览已暂停")
        image, state = preview, dict(sample)
        epoch = vision_epoch
    return Response(json.dumps({"image": base64.b64encode(image).decode(), "camera": state, "epoch": epoch}),
                    media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/calibration/frame")
def frame():
    with lock:
        sample = camera_view()
        image = preview if sample.get("ready") and not sample.get("vision_paused") else None
        private = sample.get("privacy_active")
    if image is None:
        raise HTTPException(503, "隐私保护期间暂停预览" if private else "画面暂不可用")
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/settings")
def settings(config: Settings):
    with store.database() as db:
        db.executemany("UPDATE settings SET value=? WHERE key=?",
                       [(v, k) for k, v in config.model_dump().items()])
    return {"ok": True}


@app.post("/api/calibration/{point}")
def calibrate(point: str):
    if point not in {"low", "high"}:
        raise HTTPException(404)
    with lock:
        position = camera_view().get("desk_edge")
        if position is None:
            raise HTTPException(409, "当前未找到足够清晰的桌沿，请缩小桌沿区域")
        config = json.loads(CAL.read_text()) if CAL.exists() else {}
        config[point] = position
        other = "high" if point == "low" else "low"
        if other in config and abs(config[other] - position) < .04:
            raise HTTPException(409, "与另一档的位置太近，请检查取景和桌面高度")
        save_calibration(config)
    return {point: position}


@app.post("/api/query")
def query(item: Query):
    try:
        return {"answer": respond(item.text, str(item.id), item.save)}
    except (OSError, ValueError) as error:
        raise HTTPException(503, str(error)) from error


@app.post("/api/voice/start")
def record():
    try:
        voice.start()
    except (OSError, ValueError) as error:
        raise HTTPException(409, str(error)) from error
    return {"recording": True}


@app.post("/api/voice/stop")
def answer(item: Record):
    try:
        return voice.stop(str(item.id), item.save)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        LOG.exception("Voice request failed")
        raise HTTPException(503, "本地语音处理失败，请查看日志") from error
