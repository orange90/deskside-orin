"""Measure Cosmos on operator-annotated live camera frames, with no media on disk."""
import base64
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen

from PIL import Image

from assistant import describe_desk_frames
from measure import percentile

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8765"


def camera_image():
    with urlopen(BASE + "/api/calibration/frame", timeout=5) as response:
        data = response.read()
    configuration = ROOT / "calibration.json"
    settings = json.loads(configuration.read_text()) if configuration.exists() else {}
    roi = settings.get("cosmos_roi") or settings.get("person_roi")
    with Image.open(BytesIO(data)) as original:
        image = original
        if roi:
            width, height = image.size
            image = image.crop(tuple(int(value * scale)
                                     for value, scale in zip(roi, (width, height, width, height))))
        image.thumbnail((640, 640))
        resized = BytesIO()
        image.save(resized, format="JPEG")
        encoded = base64.b64encode(resized.getvalue()).decode()
        return image.size, {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + encoded,
        }}


def observe(expected):
    size = None
    start = time.monotonic()
    try:
        size, frame = camera_image()
        start = time.monotonic()
        observed, _ = describe_desk_frames([frame], timeout=12)
        error = None
    except (OSError, ValueError, KeyError) as failure:
        observed, error = None, type(failure).__name__
    return {"expected": expected, "observed": observed, "size": size,
            "seconds": round(time.monotonic() - start, 3), "error": error}


def summarize(rows, planned):
    return {
        "planned": planned, "attempted": len(rows),
        "matched": sum(row["expected"] == row["observed"] for row in rows),
        "unknown": sum(row["observed"] == "unknown" for row in rows),
        "errors": sum(row["error"] is not None for row in rows),
        "single_frame_p95_seconds": percentile(
            [row["seconds"] for row in rows if row["error"] is None], .95),
        "requests": rows,
    }


def main():
    cases = (
        ("sitting", "正常坐在桌前。就位后按回车。"),
        ("standing", "站在桌前使用电脑。就位后按回车。"),
        ("away", "按回车后 5 秒内离开画面，保持 10 秒。"),
        ("sitting", "回来坐好，身体前倾。就位后按回车。"),
        ("standing", "站在桌前，转身露出斜后方。就位后按回车。"),
        ("away", "按回车后 5 秒内从另一侧离开画面，保持 10 秒。"),
    )
    rows = []
    attempted = 0
    try:
        for trial in range(1, 3):
            for expected, instruction in cases:
                input(f"第 {trial}/2 轮。{instruction}")
                time.sleep(5 if expected == "away" else .5)
                row = {"trial": trial, **observe(expected)}
                attempted += 1
                confirmed = input("刚才取帧时实际动作符合提示吗？确认输入 y，否则直接回车。").strip().lower() == "y"
                if confirmed:
                    rows.append(row)
                    print(f"结果 {row['observed'] or row['error']}，请求耗时 {row['seconds']:.3f} 秒。")
    finally:
        result = summarize(rows, 2 * len(cases))
        result["unconfirmed_requests"] = attempted - len(rows)
        path = ROOT / "logs" / f"field-cosmos-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"验收结果保存在本机 {path}。")


if __name__ == "__main__":
    main()
