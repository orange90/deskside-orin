"""Operator-guided desk and posture checks. Keep labels and timing on the Orin."""
import argparse
import json
from pathlib import Path
from runpy import run_path
import sqlite3
import time
from urllib.request import urlopen

from measure import percentile

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765/api/state"
POSE = (
    ("sit", "正常坐着使用电脑", 300),
    ("sit", "向桌面前倾或后仰，保持坐在椅子上", 180),
    ("sit", "轻转椅子和调整坐姿，保持坐着", 180),
    ("stand", "站起来，站着使用电脑", 420),
    ("stand", "保持站立，同时短时遮挡肩部和腿部", 60),
    ("stand", "去除遮挡，继续站着", 240),
    ("away", "走到镜头外，等终端提示阶段结束后再回来", 180),
    ("sit", "回来坐好并正常使用电脑", 240),
)
COSMOS_COUNTS = (2, 2, 2, 2, 0, 1, 2, 1)


def camera():
    with urlopen(URL, timeout=4) as response:
        return json.load(response)["camera"]


def wait_for(seconds, field, expected):
    until = time.monotonic() + seconds
    first_hit = None
    good = total = 0
    while time.monotonic() < until:
        state = camera()
        correct = state.get(field) == expected
        good += correct
        total += 1
        if correct and first_hit is None:
            first_hit = time.monotonic()
        time.sleep(.2)
    return first_hit, good, total


def desk_trial():
    results = []
    for cycle in range(1, 4):
        for target, description in (("low", "降至常用低档"), ("high", "升至常用高档")):
            input(f"第 {cycle}/3 轮。{description}，桌面停稳时按回车。")
            start = time.monotonic()
            hit, good, total = wait_for(4, "desk", target)
            results.append({"cycle": cycle, "expected": target, "samples": total,
                            "matching": good, "first_match_seconds": round(hit - start, 3) if hit else None})
    input("保持高档。按回车后离开镜头，等 12 秒再回到屏幕。")
    time.sleep(7)
    _, away, total = wait_for(5, "person", "away")
    return {"checks": results, "high_desk_away_samples": away, "high_desk_total_samples": total}


def duration_window(start, end, rows, state):
    return sum(max(0, min(end, hi) - max(start, lo)) for name, lo, hi in rows if name == state)


def pose_trial(path, with_cosmos=False):
    phases, snapshots, transition_delays = [], [], []
    cosmos = run_path(str(ROOT / "cosmos-field-check.py")) if with_cosmos else None
    cosmos_rows = []
    progress = path.with_suffix(".progress.jsonl")
    expected_transitions = 0
    observed_transitions = 0
    matched_at_first_sample = 0
    for phase, (expected, description, hold) in enumerate(POSE, 1):
        grace = 5 if expected == "away" else 0
        prompt = "按回车后在 5 秒内离开画面" if grace else "动作就位时按回车"
        input(f"第 {phase}/{len(POSE)} 段。{description}。{prompt}，保持约 {hold // 60} 分钟。")
        start, clock = time.time() + grace, time.monotonic()
        until = clock + grace + hold
        matched_at = None
        matched_initially = False
        first_sample = True
        count = COSMOS_COUNTS[phase-1] if cosmos else 0
        checks = [clock + grace + hold * i / (count + 1) for i in range(count, 0, -1)]
        with progress.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"phase": phase, "expected": expected, "start": start}) + "\n")
        while time.monotonic() < until:
            if checks and time.monotonic() >= checks[-1]:
                checks.pop()
                label = {"sit": "sitting", "stand": "standing", "away": "away"}[expected]
                cosmos_rows.append({"phase": phase, **cosmos["observe"](label)})
            status = camera()
            usable = status["ready"] and not status.get("vision_paused")
            if usable and "frame_count" in status:
                snapshots.append((status["frame_count"], status["frame_clock"]))
            if usable and status.get("person") == expected and matched_at is None:
                matched_at = time.monotonic()
                matched_initially = first_sample
            first_sample = False
            time.sleep(.25)
        end = time.time()
        confirmed = input("阶段结束。这一段确实按提示保持了该动作吗？确认输入 y，否则直接回车。").strip().lower() == "y"
        if confirmed and phases and phases[-1]["operator_confirmed"] and expected != phases[-1]["expected"]:
            expected_transitions += 1
            observed_transitions += matched_at is not None
            if matched_initially:
                matched_at_first_sample += 1
            elif matched_at is not None and expected != "away":
                transition_delays.append(round(matched_at - clock, 3))
        phases.append({"phase": phase, "expected": expected, "start": start, "end": end,
                       "operator_confirmed": confirmed,
                       "matched_on_first_sample": matched_initially,
                       "mark_reference": "leave_prompt" if expected == "away" else "operator_ready",
                       "matched_at_seconds": round(matched_at - clock, 3) if matched_at else None})
        with progress.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"phase": phase, **phases[-1]}) + "\n")
        print("阶段结束。")
    connection = sqlite3.connect(ROOT / "data.sqlite3")
    try:
        rows = connection.execute(
            "SELECT state,start,end FROM intervals WHERE kind='person' AND end>=? AND start<=?",
            (phases[0]["start"], phases[-1]["end"])
        ).fetchall()
        presence_rows = connection.execute(
            "SELECT state,start,end FROM intervals WHERE kind='presence' AND end>=? AND start<=?",
            (phases[0]["start"], phases[-1]["end"])
        ).fetchall()
        reminders = connection.execute(
            "SELECT at FROM notes WHERE kind='reminder' AND body=? AND at BETWEEN ? AND ?",
            ("已经坐了一段时间，可以起身活动一下。", phases[0]["start"], phases[-1]["end"])
        ).fetchall()
    finally:
        connection.close()
    verified = [p for p in phases if p["operator_confirmed"]]
    total = sum(p["end"] - p["start"] for p in verified)
    labels = {}
    for label in ("sit", "stand", "away", "unknown"):
        observed = sum(duration_window(p["start"], p["end"], rows, label) for p in verified)
        expected = sum(p["end"] - p["start"] for p in verified if p["expected"] == label)
        labels[label] = {
            "expected_seconds": round(expected, 2), "observed_seconds": round(observed, 2),
            "error_fraction": round(abs(observed - expected) / expected, 4) if expected else None,
        }
    differences = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(snapshots, snapshots[1:])
                   if 0 < b[1] - a[1] < 1 and b[0] >= a[0]]
    misreminders = sum(
        any(p["start"] <= at < p["end"] and p["expected"] in {"stand", "away"} for p in verified)
        for (at,) in reminders
    )
    expected_presence = sum(p["end"] - p["start"] for p in verified if p["expected"] != "away")
    observed_presence = sum(duration_window(p["start"], p["end"], presence_rows, "present") for p in verified)
    result = {"annotated_seconds": round(total, 2), "confirmed_phases": len(verified), "phases": phases,
              "ground_truth_complete": len(verified) == len(POSE), "durations": labels,
              "presence_seconds": {"expected": round(expected_presence, 2),
                                   "observed": round(observed_presence, 2)},
            "unknown_fraction": round(labels["unknown"]["observed_seconds"] / total, 4) if total else None,
            "sample_fps": round(sum(n for n, _ in differences) / sum(t for _, t in differences), 2)
                          if differences else None,
            "reminders_during_observed_stand_or_away": misreminders,
            "expected_transition_count": expected_transitions,
            "observed_transition_count": observed_transitions,
            "transition_already_matched_count": matched_at_first_sample,
            "timed_transition_count": len(transition_delays),
            "operator_mark_to_state_p95_seconds": percentile(transition_delays, .95)}
    if cosmos:
        confirmed_phases = {p["phase"] for p in verified}
        checked = [r for r in cosmos_rows if r["phase"] in confirmed_phases]
        result["cosmos"] = cosmos["summarize"](checked, sum(COSMOS_COUNTS))
        result["cosmos"]["unconfirmed_requests"] = len(cosmos_rows) - len(checked)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("desk", "pose"))
    parser.add_argument("--with-cosmos", action="store_true", help="同一轮姿态测试中同时记录 Cosmos 单帧结果")
    args = parser.parse_args()
    if args.mode == "desk":
        path = ROOT / "calibration.json"
        calibration = json.loads(path.read_text()) if path.exists() else {}
        if not {"desk_roi", "low", "high"} <= calibration.keys():
            parser.error("先在看板选择桌沿并记录高低档")
    status = camera()
    if not status["ready"]:
        parser.error("摄像头未就绪")
    if args.mode == "pose" and status.get("vision_paused"):
        parser.error("当前画面过暗。开灯并等待看板恢复检测后再开始姿态标注")
    path = ROOT / "logs" / f"field-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    result = desk_trial() if args.mode == "desk" else pose_trial(path, args.with_cosmos)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"结果保存在本机 {path}。")


if __name__ == "__main__":
    main()
