from collections import deque
from dataclasses import dataclass
from statistics import median


POSE_METHODS = frozenset({
    "keypoints", "motion", "cosmos", "multiple_people", "estimate_recent",
    "estimate_session", "estimate_pending", "estimate_joints", "estimate_default", "unknown",
})


@dataclass
class FrameLight:
    paused: bool = False
    bright_since: float = None
    dark_since: float = None
    sustained: bool = False
    last_sample: float = None

    def update(self, now, brightness):
        # A blocked read between two bright frames is not stable lighting.
        if self.last_sample is not None and now - self.last_sample >= 3:
            self.bright_since = None
        self.last_sample = now
        if brightness is None or brightness < 12:
            if self.dark_since is None:
                self.dark_since = now
            self.sustained = self.sustained or now - self.dark_since >= 3
            self.paused, self.bright_since = True, None
        elif self.paused:
            self.dark_since = None
            if self.bright_since is None:
                self.bright_since = now
            elif now - self.bright_since >= 2:
                self.paused, self.bright_since, self.sustained = False, None, False
        return self.paused


@dataclass
class StableState:
    state: str = "unknown"
    candidate: str = "unknown"
    since: float = 0

    def update(self, value, now, delay):
        if value != self.candidate:
            self.candidate, self.since = value, now
        if now - self.since >= delay:
            self.state = value
        return self.state


def desk_state(edge_y, config):
    if edge_y is None or not {"low", "high"} <= config.keys():
        return "unknown"
    gap = abs(config["high"] - config["low"])
    for state in ("low", "high"):
        if abs(edge_y - config[state]) <= gap * 0.2:
            return state
    return "transition"


def person_state(people, pose):
    if people == 0:
        return "away"
    return pose if people == 1 and pose else "unknown"


class RelativeRise:
    """Track short-term vertical motion relative to the person's own size."""
    def __init__(self):
        self.anchors = deque()
        self.pose = None
        self.pose_at = None

    def clear(self):
        self.anchors.clear()
        self.pose = None
        self.pose_at = None

    def update(self, anchor, now, camera_stationary):
        if not camera_stationary or not anchor:
            self.clear()
            return None
        x, y, scale = anchor
        if self.anchors:
            previous_time, previous_x, _, previous_scale = self.anchors[-1]
            if (not 0 < now - previous_time <= .8 or abs(x - previous_x) > .65 * scale
                    or not .67 <= scale / max(previous_scale, 1) <= 1.5):
                self.clear()
        if self.pose_at is not None and now - self.pose_at > 2.4:
            self.pose = self.pose_at = None
        while self.anchors and now - self.anchors[0][0] > 2.4:
            self.anchors.popleft()
        if len(self.anchors) >= 2:
            previous_y = median(row[2] for row in list(self.anchors)[:2])
            previous_scale = median(row[3] for row in list(self.anchors)[:2])
            shift = (y - previous_y) / max(scale, previous_scale, 1)
            last_shift = (self.anchors[-1][2] - previous_y) / max(scale, previous_scale, 1)
            if abs(shift) >= .34 and abs(last_shift) >= .34 and shift*last_shift > 0:
                self.pose = "sit" if shift > 0 else "stand"
                self.pose_at = now
                self.anchors.clear()
        self.anchors.append((now, x, y, scale))
        return self.pose


@dataclass
class SittingSession:
    seconds: float = 0
    sent: bool = False
    last: str = "unknown"
    uncertain_seconds: float = 0.

    def update(self, state, elapsed, limit):
        self.uncertain_seconds = self.uncertain_seconds + elapsed if state == "unknown" else 0.
        if self.uncertain_seconds >= 15:
            self.seconds, self.sent = 0., False
        if state in {"stand", "away"}:
            self.seconds, self.sent = 0, False
        if state == self.last == "sit":
            self.seconds += elapsed
        self.last = state
        if state == "sit" and self.seconds >= limit and not self.sent:
            self.sent = True
            return True
        return False


@dataclass
class AbsenceEvidence:
    """Absence needs three independent clear-empty reviews spanning >=6 seconds."""
    votes: int = 0
    first: float = None
    last: float = None

    def update(self, label, captured_at, empty_candidate):
        if label != "away" or not empty_candidate:
            self.votes, self.first, self.last = 0, None, None
            return False
        if self.last is not None and captured_at <= self.last:
            return False
        if self.last is None or captured_at - self.last > 10:
            self.votes, self.first = 0, captured_at
        self.votes += 1
        self.last = captured_at
        return self.votes >= 3 and captured_at - self.first >= 6


@dataclass
class PersonTrackEvidence:
    """Require a continuous detector track before starting or renewing presence."""
    first: float = None
    last: float = None
    last_supported: float = None
    supported_samples: int = 0
    samples: int = 0

    def observe(self, count, strong, joints, now):
        if not count or self.last is None or now - self.last > 0.8:
            self.first = self.last_supported = None
            self.samples = self.supported_samples = 0
        self.last = now
        if not count:
            return False
        if self.first is None:
            self.first = now
        self.samples += 1
        if strong or sum(j['confidence'] >= .55 for j in joints) >= 3:
            if self.last_supported is None or now - self.last_supported > .8:
                self.supported_samples = 0
            self.last_supported = now
            self.supported_samples += 1
        return (self.supported_samples >= 2 and self.last_supported is not None and
                now - self.last_supported <= .8 and self.samples >= 3 and now - self.first >= 1.2)


@dataclass
class PresenceContinuity:
    """A short visibility gap can retain presence, never the sitting/standing label."""
    last_positive: float = None

    def update(self, observation, now):
        if observation == "present":
            self.last_positive = now
            return "present", "observed"
        if observation == "away":
            self.last_positive = None
            return "away", "confirmed_empty"
        if self.last_positive is not None and now - self.last_positive <= 15:
            return "present", "short_occlusion"
        return "unknown", "unconfirmed"


@dataclass
class OccupiedPosture:
    """Choose a posture for an occupied workstation while preserving its provenance."""
    confirmed: str = None
    confirmed_at: float = 0.0

    joint_candidate: str = None
    joint_since: float = 0.
    joint_samples: int = 0

    @staticmethod
    def joint_hint(overlay):
        # Compare the visible shoulder-to-hip and hip-to-knee vectors. This
        # weaker cue can still be useful when ankle evidence is unavailable.
        joints = overlay.get("keypoints", [])
        votes = []
        if len(joints) == 17:
            for shoulder, hip, knee in ((5, 11, 13), (6, 12, 14)):
                a, b, c = [joints[i] for i in (shoulder, hip, knee)]
                if min(a["confidence"], b["confidence"], c["confidence"]) < .30:
                    continue
                tx, ty = b["x"]-a["x"], b["y"]-a["y"]
                lx, ly = c["x"]-b["x"], c["y"]-b["y"]
                torso, thigh = (tx*tx+ty*ty)**.5, (lx*lx+ly*ly)**.5
                if torso <= 1 or thigh < max(1., .25*torso):
                    continue
                alignment = (tx*lx+ty*ly) / (torso*thigh)
                if ly > .7*thigh and thigh > .45*torso and alignment > .8:
                    votes.append("stand")
                elif abs(ly) < .65*thigh and abs(alignment) < .6:
                    votes.append("sit")
        if votes and all(v == votes[0] for v in votes):
            return votes[0]
        return None

    def sustained_joints(self, now):
        return (self.joint_candidate is not None and self.joint_samples >= 3
                and now - self.joint_since >= .6)

    def observe_joints(self, hint, now):
        if hint is None or hint != self.joint_candidate:
            self.joint_candidate, self.joint_since, self.joint_samples = hint, now, 0
        self.joint_samples += 1
        return self.sustained_joints(now)

    def choose(self, raw, stable, source, occupied, now, overlay):
        if not occupied:
            self.confirmed = None
            return (stable, source) if raw == stable else ("unknown", None)
        if raw in {"sit", "stand"} and raw == stable:
            self.confirmed, self.confirmed_at = raw, now
            return raw, source
        hint = self.joint_hint(overlay)
        sustained = hint == self.joint_candidate and self.sustained_joints(now)
        if raw in {"sit", "stand"} and (source == "cosmos" or sustained and hint == raw):
            return raw, "estimate_pending"
        if sustained and raw not in {"sit", "stand"}:
            return hint, "estimate_joints"
        if self.confirmed and now - self.confirmed_at <= 15:
            return self.confirmed, "estimate_recent"
        if raw in {"sit", "stand"}:
            return raw, "estimate_pending"
        if hint is not None:
            return hint, "estimate_joints"
        if self.confirmed:
            return self.confirmed, "estimate_session"
        # A disclosed cold-start prior when only presence can be established.
        return "sit", "estimate_default"
