from array import array
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
from tempfile import TemporaryDirectory
import time
import wave

from assistant import ASR_HINT

suite = Path("runtime/voice-suite-regression")
cases = json.loads((suite / "cases.json").read_text())
noisy_cases = {"todo-desk", "todo-umbrella", "general-geometry", "reminder-cup"}

def normalize(text):
    text = text.lower().replace("90", "九十").replace("7", "七").replace("8", "八").replace("×", "乘")
    return re.sub(r"[^\w]", "", text)

def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]

def noise_wave(src, dst, snr, seed):
    with wave.open(str(src), "rb") as audio:
        params = audio.getparams()
        values = array("h", audio.readframes(audio.getnframes()))
    rms = math.sqrt(sum(x*x for x in values) / len(values))
    random_source = random.Random(seed)
    scale = rms * 10**(-snr/20)
    samples = array("h", (max(-32768, min(32767, round(x + scale * random_source.gauss(0, 1))))
                          for x in values))
    with wave.open(str(dst), "wb") as audio:
        audio.setparams(params)
        audio.writeframes(samples.tobytes())

def decode(path, output, pad_ms):
    begin = time.monotonic()
    subprocess.run([
        "runtime/whisper-cli", "-m", "models/ggml-small.bin", "-f", str(path),
        "-l", "zh", "-t", "4", "--prompt", ASR_HINT, "--vad", "-vm",
        "models/ggml-silero-v6.2.0.bin", "-vp", str(pad_ms), "-otxt", "-of", str(output),
    ], check=True, timeout=25, capture_output=True)
    return Path(str(output) + ".txt").read_text().strip(), time.monotonic() - begin

with TemporaryDirectory(dir="/dev/shm", prefix="orin-whisper-vad-") as folder:
    directory = Path(folder)
    collected = defaultdict(list)
    for index, case in enumerate(cases):
        name = case["id"]
        file = suite / "fixtures" / (name + ".wav")
        inputs = [("original", file)]
        if name in noisy_cases:
            for snr in (14, 8):
                noisy = directory / f"{name}-{snr}.wav"
                noise_wave(file, noisy, snr, index * 100 + snr)
                inputs.append((f"{snr}dB", noisy))
        for condition, sample in inputs:
            modes = (30, 120, 250) if index % 2 == 0 else (250, 120, 30)
            for padding in modes:
                text, seconds = decode(sample, directory / "result", padding)
                expected = normalize(case["phrase"])
                edits = distance(expected, normalize(text))
                row = {"case": name, "condition": condition, "padding_ms": padding,
                       "edits": edits, "chars": len(expected), "seconds": round(seconds, 3),
                       "query": text}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                collected[padding].append(row)
    for padding, rows in sorted(collected.items()):
        print(json.dumps({"summary": {"padding_ms": padding, "samples": len(rows),
              "total_edits": sum(row["edits"] for row in rows), "total_chars": sum(row["chars"] for row in rows),
              "exact": sum(row["edits"] == 0 for row in rows),
              "noisy_edits": sum(row["edits"] for row in rows if row["condition"] != "original"),
              "median_seconds": round(statistics.median(row["seconds"] for row in rows), 3)}}), flush=True)
