import time
from pathlib import Path

import cv2
import numpy as np

from state import RelativeRise, desk_state, person_state

ROOT = Path(__file__).resolve().parent


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
        from yolo_pose import YoloPose
        self.model = YoloPose(ROOT)
        self.model_info = self.model.info
        self.boxes = None
        self.scene = None
        self.background = None
        self.rise = RelativeRise()
        self.overlay = None

    def sample(self, frame, config):
        edge_y = desk_edge(frame, config)
        desk = desk_state(edge_y, config)
        person_roi = config.get("person_roi", [0, 0, 1, 1])
        region, offset_x, offset_y = crop(frame, person_roi)
        if self.boxes is None or self.scene != person_roi:
            self.background = None
            self.rise.clear()
        self.scene = person_roi
        # Boxes and joints are always inferred together from this exact frame.
        self.boxes, all_joints = self.model(region)
        now = time.monotonic()
        self.overlay = {'model': self.model_info['name'], 'keypoints': [], 'detections': [
            {'box': [float(v) for v in box[:4] + np.array([offset_x, offset_y, offset_x, offset_y])],
             'confidence': float(box[4])} for box in self.boxes]}
        count = len(self.boxes)
        strong = int(np.count_nonzero(self.boxes[:, 4] >= .45))
        orientation, legs, region_hint, motion = "unknown", None, None, None
        if count == 1 or strong == 1:
            selected = int(np.argmax(self.boxes[:, 4]))
            box, points = self.boxes[selected, :4], all_joints[selected]
            width = max(box[2]-box[0], (box[3]-box[1])*.75)*1.25
            # Keep the same normalized geometry thresholds used in the photo comparison.
            normalized = points[:, :2] * (192/max(float(width), 1))
            orientation = face_orientation(normalized[:, 0]*2, points[:, 2])
            legs = leg_posture(normalized, points[:, 2])
            self.overlay['keypoints'] = [
                {'x': float(p[0]+offset_x), 'y': float(p[1]+offset_y), 'confidence': float(p[2])}
                for p in points]
            region_hint = person_crop_hint(box, frame.shape, offset_x, offset_y)
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
