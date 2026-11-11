"""Sample Orin process health without exporting camera, audio or dialogue."""
import argparse
import json
import math
import time
from shutil import disk_usage
from pathlib import Path
from urllib.request import urlopen

SERVICES = {
    "dashboard": "http://127.0.0.1:8765/api/state",
    "qwen": "http://127.0.0.1:8081/health",
    "cosmos": "http://127.0.0.1:8082/health",
}


def sysfs_numbers(directory, pattern, divisor):
    values = {}
    for path in Path(directory).glob(pattern):
        try:
            label = (path.parent / "type").read_text().strip() if path.name == "temp" else path.parent.name
            with path.open("rb", buffering=0) as stream:
                data = stream.read(64)
            if data:
                values[label] = round(int(data) / divisor, 2)
        except (OSError, ValueError):
            continue
    return values


def sample():
    row = {"timestamp": time.time()}
    for name, url in SERVICES.items():
        try:
            with urlopen(url, timeout=2) as result:
                row[name] = result.status == 200
                if name == "dashboard":
                    camera = json.load(result)["camera"]
                    row.update(camera_ready=camera["ready"], fps_instant=camera.get("fps"),
                               frame_count=camera.get("frame_count"), frame_clock=camera.get("frame_clock"),
                               vision_paused=camera.get("vision_paused",False))
        except (OSError, KeyError, ValueError):
            row[name] = False
    counters = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
            counters[key] = int(value.split()[0]) / 1024
    row["available_mib"] = round(counters["MemAvailable"], 1)
    row["swap_used_mib"] = round(counters["SwapTotal"] - counters["SwapFree"], 1)
    row["disk_available_mib"] = round(disk_usage(Path(__file__).parent).free / 1024**2, 1)
    row["temperature_c"] = sysfs_numbers("/sys/class/thermal", "thermal_zone*/temp", 1000)
    row["gpu_mhz"] = sysfs_numbers("/sys/class/devfreq", "*.gpu/cur_freq", 1_000_000)
    return row


def percentile(values, percent):
    values = sorted(values)
    return values[max(0, math.ceil(percent * len(values)) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-models", action="store_true")
    args = parser.parse_args()
    initial = sample()
    if args.require_models and not all(initial.get(key) for key in ("dashboard", "qwen", "cosmos", "camera_ready")):
        parser.error("dashboard, camera, Qwen and Cosmos must all be ready")
    rows = []
    start = time.monotonic()
    with args.output.open("x") as log:
        while not rows or time.monotonic() - start < args.seconds:
            row = sample()
            previous = rows[-1] if rows else initial
            count, before = row.get("frame_count"), previous.get("frame_count")
            clock, before_clock = row.get("frame_clock"), previous.get("frame_clock")
            if (count is not None and before is not None and clock is not None and before_clock is not None
                    and count >= before and clock > before_clock):
                row["fps"] = round((count - before) / (clock - before_clock), 2)
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            time.sleep(min(args.interval, max(0, args.seconds - (time.monotonic() - start))))
    values = [row["fps"] for row in rows if row.get("fps") is not None]
    print(json.dumps({
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "samples": len(rows),
        "ready_samples": {key: sum(bool(row.get(key)) for row in rows)
                          for key in ("dashboard", "qwen", "cosmos", "camera_ready")},
        "minimum_available_mib": min(row["available_mib"] for row in rows),
        "maximum_swap_mib": max(row["swap_used_mib"] for row in rows),
        "minimum_disk_available_mib": min(row["disk_available_mib"] for row in rows),
        "vision_pause_samples": sum(row.get("vision_paused", False) for row in rows),
        "peak_temperature_c": {
            key: max(row["temperature_c"][key] for row in rows if key in row["temperature_c"])
            for key in sorted({key for row in rows for key in row["temperature_c"]})
        },
        "fps_p05": percentile(values, .05),
        "fps_median": percentile(values, .5),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
