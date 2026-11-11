from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from runpy import run_path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
pose = run_path(str(root / "field-check.py"))
clock = [0.0]
pose_cases = (
    ("sit", ["sit"] * 4, "y"),
    ("stand", ["stand"] * 4, "y"),
    ("away", ["stand"] + ["away"] * 23, "y"),
    ("sit", ["away", "sit", "sit", "sit"], "y"),
    ("stand", ["sit", "stand", "stand", "stand"], ""),
)
phases = tuple((expected, expected, 1) for expected, _, _ in pose_cases)
state = iter([value for _, readings, _ in pose_cases for value in readings])
answers = iter([answer for _, _, confirmation in pose_cases for answer in ("", confirmation)])

def fake_camera():
    return {"ready": True, "person": next(state), "frame_clock": clock[0], "frame_count": int(clock[0] * 8)}

def advance(seconds):
    clock[0] += seconds

with TemporaryDirectory(dir="/dev/shm") as tmp:
    tmp = Path(tmp)
    with sqlite3.connect(tmp / "data.sqlite3") as db:
        db.execute("CREATE TABLE intervals (kind,state,start,end)")
        db.execute("CREATE TABLE notes (kind,body,at)")
    method = pose["pose_trial"]
    with patch.dict(method.__globals__, ROOT=tmp, POSE=phases, camera=fake_camera), \
         patch("time.monotonic", side_effect=lambda: clock[0]), \
         patch("time.time", side_effect=lambda: 1600000000 + clock[0]), \
         patch("time.sleep", side_effect=advance), patch("builtins.input", side_effect=lambda _: next(answers)), \
         redirect_stdout(StringIO()):
        report = method(tmp / "field-pose.json")
    assert report["confirmed_phases"] == 4
    assert report["expected_transition_count"] == report["observed_transition_count"] == 3
    assert report["transition_already_matched_count"] == report["timed_transition_count"] == 1
    assert report["operator_mark_to_state_p95_seconds"] == .25
    assert report["phases"][2]["start"] == 1600000007 and report["phases"][2]["mark_reference"] == "leave_prompt"
    assert report["durations"]["away"]["expected_seconds"] == 1
    print("POSE_CONFIRMED_ONLY_AND_CENSORED_TRANSITIONS_OK")
    print("pose timing:", json.dumps({key: report[key] for key in ("expected_transition_count", "observed_transition_count", "transition_already_matched_count", "timed_transition_count", "operator_mark_to_state_p95_seconds")}))
    print("away grace seconds: 5")

    cosmos = run_path(str(root / "cosmos-field-check.py"))
    (tmp / "logs").mkdir()
    cosmos_main = cosmos["main"]
    confirmations = iter(["", "y"] + ["", ""] * 11)
    def fake_observe(expected):
        return {"expected": expected, "observed": expected, "error": None, "seconds": .25, "size": [640, 360]}
    with patch.dict(cosmos_main.__globals__, ROOT=tmp, observe=fake_observe), patch("time.sleep"), \
         patch("builtins.input", side_effect=lambda _: next(confirmations)), redirect_stdout(StringIO()):
        cosmos_main()
    result = json.loads(next((tmp / "logs").glob("field-cosmos-*.json")).read_text())
    assert result["attempted"] == result["matched"] == 1
    assert result["unconfirmed_requests"] == 11
    print("COSMOS_CONFIRMED_ONLY_OK")
