from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass
class FrameLight:
    paused: bool = False
    bright_since: float = None
    dark_since: float = None
    sustained: bool = False

    def update(self, now, brightness):
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

    def clear(self):
        self.anchors.clear()
        self.pose = None

    def update(self, anchor, now, camera_stationary):
        if not camera_stationary or not anchor:
            self.clear()
            return None
        x, y, scale = anchor
        while self.anchors and now - self.anchors[0][0] > 2.4:
            self.anchors.popleft()
        if self.anchors and abs(x - self.anchors[-1][1]) > .65 * scale:
            self.clear()
        if len(self.anchors) >= 2:
            previous_y = median(row[2] for row in list(self.anchors)[:2])
            previous_scale = median(row[3] for row in list(self.anchors)[:2])
            shift = (y - previous_y) / max(scale, previous_scale, 1)
            if abs(shift) >= .34:
                self.pose = "sit" if shift > 0 else "stand"
                self.anchors.clear()
        self.anchors.append((now, x, y, scale))
        return self.pose


@dataclass
class SittingSession:
    seconds: float = 0
    sent: bool = False
    last: str = "unknown"

    def update(self, state, elapsed, limit):
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

    def choose(self, raw, stable, source, occupied, now, overlay):
        if not occupied:
            self.confirmed = None
            return (stable, source) if raw == stable else ("unknown", None)
        if raw in {"sit", "stand"} and raw == stable:
            self.confirmed, self.confirmed_at = raw, now
            return raw, source
        if self.confirmed and now - self.confirmed_at <= 15:
            return self.confirmed, "estimate_recent"
        if raw in {"sit", "stand"}:
            return raw, "estimate_pending"

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
                if min(torso, thigh) > 1:
                    votes.append(ly > .7*thigh and thigh > .45*torso and
                                 tx*lx+ty*ly > .8*torso*thigh)
        if votes and all(v == votes[0] for v in votes):
            return ("stand" if votes[0] else "sit"), "estimate_joints"
        if self.confirmed:
            return self.confirmed, "estimate_session"
        # A disclosed cold-start prior when only presence can be established.
        return "sit", "estimate_default"
