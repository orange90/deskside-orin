import time
from pathlib import Path

import cv2
import numpy as np

from state import RelativeRise, desk_state, person_state

ROOT = Path(__file__).resolve().parent


class Model:
    def __init__(self, name):
        from trt_model import TensorRT
        shape = (1, 3, 640, 640) if name == "rtmdet" else (1, 3, 256, 192)
        self.backend = TensorRT(ROOT / "models" / f"{name}.plan", shape)

    def __call__(self, image, mean, std):
        image = (image.astype(np.float32) - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        return self.backend(tensor)


def crop(frame, roi):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(value * scale) for value, scale in zip(roi, (w, h, w, h)))
    return frame[y1:y2, x1:x2], x1, y1


def desk_edge(frame, config):
    roi = config.get("desk_roi")
    if roi is None:
        return None
    region, _, offset = crop(frame, roi)
    if not region.size:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 100)
    minimum_width = max(30, round(region.shape[1] * .4))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(18, minimum_width // 2),
                           minLineLength=minimum_width, maxLineGap=10)
    if lines is None:
        return None

    horizontal = []
    for x1, y1, x2, y2 in lines[:, 0]:
        width = abs(x2 - x1)
        if width >= minimum_width and abs(y2 - y1) <= .2 * width:
            height = ((y1 + y2) / 2 + offset) / frame.shape[0]
            xs = np.rint(np.linspace(x1, x2, 16)).astype(int)
            ys = np.rint(np.linspace(y1, y2, 16)).astype(int)
            neighborhood = gray[np.clip(ys[None, :] + np.arange(-3, 4)[:, None],
                                        0, gray.shape[0] - 1), xs]
            darkness = 1 - float(np.percentile(neighborhood, 25)) / 255
            horizontal.append((width * darkness**2, height))
    if not horizontal:
        return None

    horizontal.sort(reverse=True)
    score, height = horizontal[0]
    competitors = (other_score for other_score, other_height in horizontal[1:]
                   if abs(other_height - height) >= .035)
    if max(competitors, default=0) >= score * .80:
        return None
    return float(height)


def person_crop_hint(box, shape, offset_x, offset_y):
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = shape[:2]
    margin = .12 * max(x2 - x1, y2 - y1)
    roi = [
        max(0, (x1 + offset_x - margin) / w),
        max(0, (y1 + offset_y - margin) / h),
        min(1, (x2 + offset_x + margin) / w),
        min(1, (y2 + offset_y + margin) / h),
    ]
    return roi if roi[2] > roi[0] and roi[3] > roi[1] else None


def face_orientation(x, scores):
    """Report image-relative head cues only when nose and both eyes are visible."""
    if min(scores[:3]) < .50:
        return "unknown"
    nose, eye_a, eye_b = (int(value) for value in x[:3])
    eye_span = abs(eye_a - eye_b)
    if eye_span < 5:
        return "unknown"
    offset = (nose - (eye_a + eye_b) / 2) / eye_span
    if offset < -.35:
        return "image_left"
    if offset > .35:
        return "image_right"
    return "toward_camera"



def leg_posture(points, scores):
    """Classify from reliable relative joint angles, independent of frame height."""
    votes = []
    for shoulder, hip_id, knee_id, ankle_id in ((5, 11, 13, 15), (6, 12, 14, 16)):
        if min(scores[hip_id], scores[knee_id]) < .55:
            continue
        indices = [hip_id, knee_id, ankle_id]
        hip, knee, ankle = points[indices]
        if scores[shoulder] >= .55:
            torso, upper_leg = hip - points[shoulder], knee - hip
            denominator = np.linalg.norm(torso) * np.linalg.norm(upper_leg)
            if denominator >= 144 and abs(float(np.dot(torso, upper_leg) / denominator)) < .60:
                votes.append("sit")
                continue
        if scores[ankle_id] < .55:
            continue
        thigh, shin = hip - knee, ankle - knee
        lengths = np.linalg.norm(thigh) * np.linalg.norm(shin)
        if min(np.linalg.norm(thigh), np.linalg.norm(shin)) < 12:
            continue
        cosine = float(np.dot(thigh, shin) / lengths)
        if cosine < -.87 and ankle[1] - hip[1] > .60 * np.linalg.norm(ankle - hip):
            votes.append("stand")
        elif -.57 < cosine < .57 and abs(thigh[1]) < abs(thigh[0]) and shin[1] > 10:
            votes.append("sit")
    return votes[0] if votes and all(v == votes[0] for v in votes) else None


def background_stable(previous, current, shape, box):
    if previous is None:
        return False
    height, width = current.shape
    frame_h, frame_w = shape[:2]
    left, top, right, bottom = np.rint(box * [width/frame_w, height/frame_h,
                                               width/frame_w, height/frame_h]).astype(int)
    mask = np.full_like(previous, 255)
    mask[max(0, top-7):min(height, bottom+7), max(0, left-8):min(width, right+8)] = 0
    features = cv2.goodFeaturesToTrack(previous, 40, .025, 5, mask=mask)
    if features is None or len(features) < 6:
        return False
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, features, None)
    if tracked is None:
        return False
    distances = np.linalg.norm(tracked[status.ravel() == 1] - features[status.ravel() == 1], axis=2)
    # The chair and desk may move with the subject. Require a group of
    # stationary background points, including when these objects dominate.
    fixed = int(np.count_nonzero(distances < 1.6))
    return fixed >= 8 and fixed * 4 >= len(distances)


class Vision:
    def __init__(self):
        self.detector = Model("rtmdet")
        self.pose = Model("rtmpose")
        self.boxes = None
        self.scene = None
        self.detect_at = 0
        self.background = None
        self.rise = RelativeRise()
        self.overlay = None

    def people(self, frame):
        h, w = frame.shape[:2]
        ratio = min(640 / h, 640 / w)
        resized = cv2.resize(frame, (round(w * ratio), round(h * ratio)))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        raw_boxes, scores = self.detector(canvas, (103.53, 116.28, 123.675),
                                          (57.375, 57.12, 58.395))
        scores = scores[0, :, 0]
        candidates = scores >= .28
        boxes = raw_boxes[0, candidates].astype(np.float32)
        scores = scores[candidates].astype(np.float32)
        if not boxes.size:
            return np.empty((0, 5), np.float32)
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        keep = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), .27, .45),
                          dtype=np.int64).reshape(-1)
        return np.column_stack((boxes[keep] / ratio, scores[keep]))

    def body_cues(self, image, box):
        x1, y1, x2, y2 = box
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        width = max(x2 - x1, (y2 - y1) * .75) * 1.25
        if width <= 0:
            return "unknown", None, None
        scale = np.array([width, width / .75])
        origin = center - scale / 2
        affine = np.array([[192 / scale[0], 0, -origin[0] * 192 / scale[0]],
                           [0, 256 / scale[1], -origin[1] * 256 / scale[1]]])
        pixels = cv2.warpAffine(image, affine, (192, 256))
        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        x, y = self.pose(rgb, (123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
        scores = np.minimum(x[0].max(axis=-1), y[0].max(axis=-1))
        x_index, y_index = x[0].argmax(axis=-1), y[0].argmax(axis=-1)
        orientation = face_orientation(x_index, scores)
        legs = leg_posture(np.column_stack((x_index, y_index)) / 2, scores)
        coordinates = origin + (np.column_stack((x_index, y_index))/2) * scale / np.array([192, 256])
        joints = [{'x': float(xy[0]), 'y': float(xy[1]), 'confidence': float(score)}
                  for xy, score in zip(coordinates, scores)]
        return orientation, legs, joints

    def sample(self, frame, config):
        edge_y = desk_edge(frame, config)
        desk = desk_state(edge_y, config)
        person_roi = config.get("person_roi", [0, 0, 1, 1])
        region, offset_x, offset_y = crop(frame, person_roi)
        now = time.monotonic()
        scene = person_roi
        fresh = self.boxes is None or self.scene != scene or now >= self.detect_at
        if fresh:
            if self.scene != scene or self.boxes is None:
                self.background = None
                self.rise.clear()
            self.scene = scene
            self.boxes = self.people(region)
            self.detect_at = now + .45
        self.overlay = {'keypoints': [], 'detections': [
            {'box': [float(v) for v in box[:4] + np.array([offset_x, offset_y, offset_x, offset_y])],
             'confidence': float(box[4])} for box in self.boxes]}
        count = len(self.boxes)
        strong = int(np.count_nonzero(self.boxes[:, 4] >= .45))
        orientation, legs, region_hint, motion = "unknown", None, None, None
        if count == 1 or strong == 1:
            box = self.boxes[int(np.argmax(self.boxes[:, 4])), :4]
            orientation, legs, joints = self.body_cues(region, box)
            if joints:
                self.overlay['keypoints'] = [{**p, 'x': p['x'] + offset_x, 'y': p['y'] + offset_y} for p in joints]
            region_hint = person_crop_hint(box, frame.shape, offset_x, offset_y)
            if fresh:
                gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
                full_box = box + (offset_x, offset_y, offset_x, offset_y)
                x1, y1, x2, y2 = full_box
                steady = background_stable(self.background, gray, frame.shape, full_box)
                self.rise.update(((x1+x2)/2, y1, x2-x1), now, steady)
                self.background = gray
            motion = self.rise.pose
        else:
            self.rise.clear()
        pose = legs or motion
        person = person_state(1 if strong == 1 else count, pose)
        source = "keypoints" if legs else "motion" if motion else None
        return desk, person, edge_y, count, strong, orientation, region_hint, source
