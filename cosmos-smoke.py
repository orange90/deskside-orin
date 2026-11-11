"""Check one- and three-image payloads using generated pictures stored in RAM."""
import base64
import io
import json
import time

from PIL import Image, ImageDraw

from assistant import describe_desk_frames


def image(x, posture="standing"):
    frame = Image.new("RGB", (320, 240), "#f1e9d2")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((185, 139, 306, 155), fill="#5b392a")
    draw.rectangle((194, 155, 207, 220), fill="#5b392a")
    draw.rectangle((270, 90, 306, 139), fill="#293a47")
    if posture == "standing":
        draw.ellipse((x - 18, 63, x + 18, 99), fill="#d3986a")
        draw.line((x, 102, x, 168, x - 14, 219), fill="#375d91", width=17)
        draw.line((x, 168, x + 15, 219), fill="#375d91", width=17)
        draw.line((x, 119, x + 22, 146), fill="#375d91", width=14)
    elif posture == "sitting":
        draw.line((x-23, 117, x-23, 188, x+20, 188), fill="#777477", width=10)
        draw.line((x-18, 185, x-18, 221), fill="#777477", width=8)
        draw.ellipse((x-15, 101, x+15, 131), fill="#d3986a")
        draw.line((x, 135, x, 181, x+29, 181, x+29, 220), fill="#375d91", width=15)
        draw.line((x, 143, x+27, 146), fill="#375d91", width=12)
    data = io.BytesIO()
    frame.save(data, format="JPEG")
    encoded = base64.b64encode(data.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def main():
    cases = [
        ("standing", [image(174)]),
        ("sitting", [image(153, "sitting")]),
        ("away", [image(0, "away")]),
        ("standing", [image(x) for x in (174, 137, 93)]),
    ]
    results = []
    for expected, frames in cases:
        begin = time.monotonic()
        try:
            label, summary = describe_desk_frames(frames, timeout=120)
            row = {"label": label, "summary": summary}
        except (OSError, ValueError, KeyError) as error:
            row = {"error": str(error)}
        row["passed"] = row.get("label") == expected
        results.append(row["passed"])
        print(json.dumps({"expected": expected, "frames": len(frames), "size": [320, 240],
                          "seconds": round(time.monotonic() - begin, 3), **row},
                         ensure_ascii=False), flush=True)
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
