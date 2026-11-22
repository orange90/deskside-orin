import base64
import json
import logging
import os
import queue
import signal
import sqlite3
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, StrictBool, field_validator

import store
import focus_sessions
import focus_handoffs
from idea_search import search_ideas
import field_trials
import privacy_pause
import reminder_queue
import voice_replacements
import interval_corrections
import activity_evidence
import model_registry
from dashboard_data import activity_range, activity_week, captured_idea_day, daily_ideas_markdown, dashboard_snapshot, weekly_markdown
from telemetry import monitor
from assistant import Voice, describe_desk_frames, idea_body, respond
from state import (POSE_METHODS, AbsenceEvidence, FrameLight, OccupiedPosture, PersonTrackEvidence,
                   PresenceContinuity, SittingSession, StableState)

LOG = logging.getLogger("uvicorn.error")
ROOT = store.ROOT
CAL = ROOT / "calibration.json"
voice = Voice()
stop = threading.Event()
events = queue.Queue(maxsize=1)
lock = threading.RLock()
capture_released = threading.Condition(lock)
privacy_control = threading.Lock()
# The capture generation changes on both edges, including a quick pause/resume
# that the thread never saw between reads. Only the capture thread releases V4L2.
capture_generation = 0
device_generation = None
manual_pause = None
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


def publish(frame, *, generation=None, **values):
    global latest, preview, last_publication
    with lock:
        if manual_pause is not None or generation is not None and generation != capture_generation:
            return
        if latest.get("ready") and not latest.get("vision_paused") and (
                not values.get("ready") or values.get("vision_paused")):
            invalidate_scene_locked()
        last_publication = time.monotonic()
        latest = values
        preview = frame


def sync_privacy(action=privacy_pause.snapshot, *args, confirm=False, startup=False):
    global manual_pause, capture_generation, latest, preview
    # Serialize durable changes with installation of their in-memory gate.
    # Never hold the publication lock across camera reads or model calls.
    with privacy_control:
        with capture_released:
            status = action(*args)
            active = status['active']
            previous = manual_pause
            manual_pause = active
            # Extending the same pause only moves its deadline. The camera
            # stays released and the existing privacy sample clock continues.
            if startup or (previous or {}).get('id') != (active or {}).get('id'):
                capture_generation += 1
                invalidate_scene_locked()
                preview = None
                latest = {"ready": False}
                capture_released.notify_all()
                capture_clock.boundary(previous, active, status['server_now'], startup)
            if confirm and active and not capture_released.wait_for(lambda: device_generation is None, timeout=2):
                raise HTTPException(503, "已屏蔽摄像头画面，正在等待设备释放。请重试确认。")
        return status


def queue_scene(frames, reply, mode, deadline, generation):
    with lock:
        if manual_pause is not None or generation is not None and generation != capture_generation:
            return False
        try:
            events.put_nowait((frames, reply, vision_epoch, mode, deadline))
        except queue.Full:
            return False
        return True


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

    def update(self, frame, now, *, empty_candidate, interval, generation=None):
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
            deadline = now + (max(4, interval + 2) if empty_candidate else 7)
            if queue_scene([frame()], reply, "occupancy" if empty_candidate else "person", deadline, generation):
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


PRIVATE_STATES = {"person": "privacy", "presence": "privacy", "desk": "unknown", "pose_method": "unknown"}


class CaptureClock:
    def __init__(self):
        self.last = 0.
        self.wall = self.generation = None

    def write(self, generation, states, wall, monotonic, continuous):
        with lock:
            if generation == capture_generation:
                store.persist_sample(states, wall, continuous)
                self.last = monotonic
                self.wall, self.generation = wall, generation

    def boundary(self, previous, active, wall, startup):
        """Called under the publication lock after invalidating the old generation."""
        now = time.monotonic()
        recent = (not startup and self.generation == capture_generation - 1 and
                  0 <= now - self.last < 3 and 0 <= wall - self.wall <= 3)
        if previous and recent:
            end = min(wall, previous['until'])
            if end > self.wall:
                store.persist_sample(PRIVATE_STATES, end, True)
        self.last = 0.
        self.wall = self.generation = None
        if active:
            # On restart observe only from here, even if the lease started hours ago.
            self.write(capture_generation, PRIVATE_STATES, wall, now, recent)


capture_clock = CaptureClock()


def capture_frames():
    global device_generation, last_publication
    import cv2
    from vision import Vision

    cv2.setNumThreads(1)
    vision = None
    while not stop.is_set():
        with lock:
            generation, active = capture_generation, manual_pause
        if active:
            wall, now = time.time(), time.monotonic()
            if wall >= active['until']:
                sync_privacy()
                continue
            if capture_clock.generation != generation or not 0 <= now - capture_clock.last < 1:
                continuous = capture_clock.generation == generation and 0 <= now - capture_clock.last < 3
                capture_clock.write(generation, PRIVATE_STATES, wall, now, continuous)
            last_publication = now
            stop.wait(min(.5, active['until'] - wall))
            continue
        if vision is None:
            vision = Vision()
        with capture_released:
            if generation != capture_generation:
                continue
            device_generation = generation
        try:
            _capture_session(cv2, vision, generation)
        finally:
            # A fresh session must start with a new scene, subject and pose history.
            # The loaded model is reusable; Vision.sample resets its tracker when boxes is None.
            vision.boxes = vision.background = vision.last_subject_box = vision.last_subject_time = None
            vision.overlay = {}
            if hasattr(vision, "rise"):
                vision.rise.clear()
            with capture_released:
                device_generation = None
                capture_released.notify_all()


def _capture_session(cv2, vision, generation):
    posture, desk, facing = StableState(), StableState(), StableState()
    session = SittingSession()
    continuity = PresenceContinuity()
    estimate = OccupiedPosture()
    track = PersonTrackEvidence()
    history = deque(maxlen=3)
    cap = None
    last_sample = None
    frame_count = 0
    last_db = 0.
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
        while not stop.is_set() and generation == capture_generation:
            start = time.monotonic()
            if cap is None:
                cap = cv2.VideoCapture(int(os.environ.get("ORIN_CAMERA", "0")), cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if generation != capture_generation:
                    return
            ok, frame = cap.read()
            if generation != capture_generation:
                return
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
                    if generation != capture_generation:
                        return
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
                                                 empty_candidate=empty_candidate, interval=interval, generation=generation)
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
                publish(preview_data, generation=generation, ready=True, person=ps, desk=ds, desk_edge=edge_y,
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
                    with lock:
                        if generation == capture_generation:
                            store.note("reminder", "已经坐了一段时间，可以起身活动一下。")
                if region_hint:
                    last_subject_roi = region_hint
                roi = config.get("cosmos_roi") or last_subject_roi
                if not paused and not multiple and roi and (not history or now - history[-1][0] >= .75):
                    history.append((now, desk_image(frame, roi)))
                if ps != previous and previous in {"sit", "stand", "away"} and ps in {"sit", "stand", "away"} and history:
                    queue_scene([data for _, data in history], None, None, now + 8, generation)
                previous, last_sample = ps, now
            else:
                publish(None, generation=generation, ready=False, vision_paused=True, privacy_active=privacy,
                        error="摄像头不可用")
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
                # Sample the method on the same clock as the posture. Changes
                # of method alone never schedule extra frame-rate DB writes.
                method = "unknown"
                if ok and not paused and not privacy and pose_source in POSE_METHODS:
                    method = pose_source
                capture_clock.write(generation, {**states, "pose_method": method}, captured_at, now, continuous)
                last_recorded = states
                observation_broken = False
                last_db = now
            if not ok:
                with capture_released:
                    capture_released.wait_for(lambda: generation != capture_generation or stop.is_set(), timeout=2)
            stop.wait(max(0, .16 - (time.monotonic() - start)))
    finally:
        if cap is not None:
            cap.release()


def cosmos():
    while not stop.is_set():
        frames = images = None
        try:
            frames, reply, epoch, review_mode, deadline = events.get(timeout=.5)
        except queue.Empty:
            continue
        while epoch == vision_epoch and time.monotonic() < deadline and (voice.busy.is_set() or voice.recording):
            if stop.wait(.15):
                return
        if manual_pause is not None or epoch != vision_epoch or time.monotonic() >= deadline:
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
            with lock:
                if manual_pause is not None or epoch != vision_epoch or time.monotonic() >= deadline:
                    if reply is not None:
                        reply.set_result(None)
                    continue
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
        try:
            with store.database() as db:
                sit_limit = db.execute("SELECT value FROM settings WHERE key='sit_minutes'").fetchone()[0] * 60
            store.record_due_reminders()
        except sqlite3.Error:
            LOG.exception("Reminder check failed; retrying after the database is available")
            if stop.wait(5):
                return


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
    sync_privacy(startup=True)
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
    text: str = Field(min_length=1, max_length=540)
    id: uuid.UUID
    save: bool = False

    @field_validator("text")
    @classmethod
    def check_text_length(cls, text):
        if len(text) > 500:
            idea = idea_body(text)
            if idea is None:
                raise ValueError("问题最多 500 字")
            if len(idea) > 500:
                raise ValueError("灵感正文最多 500 字")
        return text


class Record(BaseModel):
    id: uuid.UUID
    save: bool = False


class Settings(BaseModel):
    stand_minutes: int = Field(ge=1, le=1440)
    sit_minutes: int = Field(ge=1, le=1440)


class AbsenceRange(BaseModel):
    start_date: str
    end_date: str
    request_id: uuid.UUID | None = None


class IntervalCorrection(BaseModel):
    id: uuid.UUID
    date: str
    start: float = Field(strict=True)
    end: float = Field(strict=True)
    state: str


class TodoCompletion(BaseModel):
    id: str = Field(strict=True, min_length=1)
    completed: StrictBool


class FocusStart(BaseModel):
    id: uuid.UUID
    minutes: int = Field(strict=True, ge=1, le=180)
    title: str | None = None
    task_id: str | None = None


class HandoffContent(BaseModel):
    body: str = Field(strict=True, min_length=1, max_length=500)


class IdeaSearch(BaseModel):
    query: str = Field(strict=True, min_length=1)
    before: str | None = Field(default=None, strict=True, max_length=512)


class VoiceReplacement(BaseModel):
    id: uuid.UUID
    heard: str = Field(strict=True)
    written: str = Field(strict=True)


class ResumeHandoff(BaseModel):
    id: uuid.UUID


class PrivacyStart(BaseModel):
    id: uuid.UUID
    minutes: int = Field(strict=True)


class PrivacyExtension(PrivacyStart):
    expected_until: float = Field(strict=True, allow_inf_nan=False)


class TrialLabel(BaseModel):
    id: uuid.UUID
    state: str


class ReminderChange(BaseModel):
    id: str = Field(strict=True, min_length=1)
    request_id: uuid.UUID
    action: str
    expected_due: float = Field(strict=True, allow_inf_nan=False)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (ROOT / "dashboard.html").read_text()


@app.get("/field-trial", response_class=HTMLResponse)
def trial_page():
    return (ROOT / "field-trial.html").read_text()


def trial_action(action, *args):
    try:
        return action(*args)
    except KeyError:
        raise HTTPException(404, "找不到该次姿态核验") from None
    except field_trials.TrialConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/field-trials")
def trial_status():
    return field_trials.snapshot()


@app.post("/api/field-trials")
def trial_start(item: TrialLabel):
    return trial_action(field_trials.start, item.id, item.state)


@app.post("/api/field-trials/{session_id}/marks")
def trial_mark(session_id: uuid.UUID, item: TrialLabel):
    return trial_action(field_trials.mark, session_id, item.id, item.state)


@app.post("/api/field-trials/{session_id}/finish")
def trial_finish(session_id: uuid.UUID):
    return trial_action(field_trials.finish, session_id)


@app.get("/api/field-trials/{session_id}/report")
def trial_report(session_id: uuid.UUID):
    return trial_action(field_trials.report, session_id)


def camera_view():
    """Published frames are immutable. A stalled capture must not look live."""
    with lock:
        sample, active = latest, manual_pause
    manual = {"manual_privacy": active is not None, "manual_until": active['until'] if active else None,
              "manual_id": active['id'] if active else None}
    if active:
        return {**manual, "ready": False, "vision_paused": True, "privacy_active": True,
                "person": "privacy", "presence": "privacy"}
    if sample.get("ready") and time.monotonic() - sample.get("frame_clock", 0) > 3:
        return {**manual, "ready": False, "stale": True, "vision_paused": True,
                "person": "unknown", "presence": "unknown", "people": None,
                "privacy_active": bool(sample.get("privacy_active")),
                "error": "摄像头没有新画面"}
    return {**sample, **manual}


@app.get("/api/camera/privacy")
def camera_privacy():
    return sync_privacy()


@app.post("/api/camera/privacy")
def pause_camera(item: PrivacyStart):
    try:
        return sync_privacy(privacy_pause.start, item.id, item.minutes, confirm=True)
    except privacy_pause.PrivacyConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/camera/privacy/{session_id}/extend")
def extend_camera_pause(session_id: uuid.UUID, item: PrivacyExtension):
    try:
        return sync_privacy(privacy_pause.extend, session_id, item.id, item.minutes,
                            item.expected_until, confirm=True)
    except KeyError:
        raise HTTPException(404, "找不到该次隐私暂停") from None
    except privacy_pause.PrivacyConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/camera/privacy/{session_id}/resume")
def resume_camera(session_id: uuid.UUID):
    try:
        return sync_privacy(privacy_pause.resume, session_id)
    except KeyError:
        raise HTTPException(404, "找不到该次隐私暂停") from None


@app.get("/api/state")
def state():
    return {**store.snapshot(), "camera": camera_view(), "recording": bool(voice.recording)}


class ModelSelection(BaseModel):
    component: str
    model: str


@app.get("/api/focus")
def focus_status():
    return focus_sessions.snapshot()


@app.post("/api/focus")
def start_focus(item: FocusStart):
    try:
        return focus_sessions.start(item.id, item.minutes, item.title, item.task_id)
    except KeyError:
        raise HTTPException(404, "找不到该待办") from None
    except focus_sessions.FocusConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/focus/{session_id}/finish")
def finish_focus(session_id: uuid.UUID):
    try:
        return focus_sessions.finish(session_id)
    except KeyError:
        raise HTTPException(404, "找不到该次专注计时") from None


def handoff_action(action, *args):
    try:
        return action(*args)
    except KeyError:
        raise HTTPException(404, "找不到这次专注或续做便签") from None
    except (focus_handoffs.HandoffConflict, focus_sessions.FocusConflict) as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/focus/handoffs")
def pending_handoffs(before: str | None = None):
    return handoff_action(focus_handoffs.pending, before)


@app.post("/api/focus/{session_id}/handoff")
def leave_handoff(session_id: uuid.UUID, item: HandoffContent):
    return handoff_action(focus_handoffs.save, session_id, item.body)


@app.post("/api/focus/handoffs/{session_id}/resume")
def resume_handoff(session_id: uuid.UUID, item: ResumeHandoff):
    return handoff_action(focus_handoffs.resume, session_id, item.id)


@app.post("/api/focus/handoffs/{session_id}/dismiss")
def dismiss_handoff(session_id: uuid.UUID):
    return handoff_action(focus_handoffs.dismiss, session_id)


@app.get("/api/todos")
def todos(completed: bool = False, before: str | None = None):
    try:
        return store.list_todos(completed, before)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/todos/completion")
def todo_completion(item: TodoCompletion):
    try:
        return store.set_todo_completed(item.id, item.completed)
    except KeyError:
        raise HTTPException(404, "找不到该待办") from None
    except store.TodoConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/reminders")
def reminder_feed(after: int = 0):
    with store.database() as db:
        rows = [dict(row) for row in db.execute(
            "SELECT id,body,at FROM notes WHERE kind='reminder' AND id>? ORDER BY id LIMIT 100",
            (max(0, after),))]
        newest = db.execute("SELECT COALESCE(MAX(id),0) FROM notes WHERE kind='reminder'").fetchone()[0]
    return {"reminders": rows, "latest_id": newest}


@app.get("/api/reminders/pending")
def pending_reminders(cursor: str | None = None):
    try:
        return reminder_queue.list_pending(cursor)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/reminders/manage")
def change_reminder(item: ReminderChange):
    try:
        return reminder_queue.change(item.id, item.request_id, item.action, item.expected_due)
    except KeyError:
        raise HTTPException(404, "找不到该提醒") from None
    except reminder_queue.ReminderConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


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


@app.get("/api/activity/evidence")
def interval_evidence(date: str, start: float, end: float):
    try:
        return activity_evidence.details(date, start, end)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/activity/range")
def range_activity(start: str, end: str):
    try:
        return activity_range(start, end)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/activity/week")
def weekly_activity(end_date: str | None = None):
    try:
        return activity_week(end_date)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/activity/week.md", response_class=Response, responses={200: {"content": {"text/markdown": {}}}})
def export_week(end_date: str | None = None):
    try:
        filename, document = weekly_markdown(end_date)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return markdown_download(filename, document)


@app.get("/api/ideas/day.md", response_class=Response, responses={200: {"content": {"text/markdown": {}}}})
def export_ideas(date: str):
    try:
        filename, document = daily_ideas_markdown(date)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return markdown_download(filename, document)


@app.post("/api/ideas/search")
def find_ideas(item: IdeaSearch):
    try:
        return JSONResponse(search_ideas(item.query, item.before), headers={"Cache-Control": "no-store"})
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def markdown_download(filename, document):
    return Response(document, media_type="text/markdown", headers={
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
    })


@app.post("/api/activity/corrections")
def correct_interval(item: IntervalCorrection):
    try:
        return interval_corrections.create(item.id, item.date, item.start, item.end, item.state)
    except interval_corrections.CorrectionConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/activity/corrections/{request_id}/undo")
def undo_interval(request_id: uuid.UUID):
    try:
        interval_corrections.undo(request_id)
    except KeyError:
        raise HTTPException(404, "找不到该次时段校正") from None
    return {"ok": True}


@app.get("/api/absences")
def absence_history(date: str):
    try:
        return {"batches": store.list_absences(date)}
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/absences")
def confirm_absence(item: AbsenceRange):
    try:
        return store.create_absence(item.start_date, item.end_date, item.request_id)
    except store.AbsenceConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/absences/{batch_id}/undo")
def remove_absence(batch_id: str):
    try:
        store.undo_absence(batch_id)
    except KeyError:
        raise HTTPException(404, "找不到该次离开校正") from None
    return {"ok": True}


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


def add_idea_date(reply, query, request_id):
    if idea_body(query):
        day = captured_idea_day(request_id)
        if day is not None:
            reply["idea_date"] = day
    return reply


@app.post("/api/query")
def query(item: Query):
    try:
        reply = {"answer": respond(item.text, str(item.id), item.save)}
        return add_idea_date(reply, item.text, item.id)
    except (OSError, ValueError) as error:
        raise HTTPException(503, str(error)) from error


@app.get("/api/voice/replacements")
def list_voice_replacements():
    return JSONResponse(voice_replacements.list_rules(), headers={"Cache-Control": "no-store"})


@app.post("/api/voice/replacements")
def save_voice_replacement(item: VoiceReplacement):
    try:
        result = voice_replacements.save_rule(str(item.id), item.heard, item.written)
    except voice_replacements.RuleConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.post("/api/voice/replacements/{rule_id}/remove")
def remove_voice_replacement(rule_id: uuid.UUID):
    return JSONResponse(voice_replacements.remove_rule(str(rule_id)), headers={"Cache-Control": "no-store"})


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
        reply = voice.stop(str(item.id), item.save)
        return add_idea_date(reply, reply["query"], item.id)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        LOG.exception("Voice request failed")
        raise HTTPException(503, "本地语音处理失败，请查看日志") from error


@app.post("/api/voice/draft")
def voice_draft():
    try:
        return voice.draft()
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        LOG.exception("Voice transcription failed")
        raise HTTPException(503, "本地语音转写失败，请查看日志") from error
