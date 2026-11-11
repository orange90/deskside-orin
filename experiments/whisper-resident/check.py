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
from urllib.request import Request, urlopen

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

def decode(path, output, model):
    boundary = "orin-whisper-fixture-boundary"
    headers = (f"--{boundary}\r\n"
               'Content-Disposition: form-data; name="file"; filename="fixture.wav"\r\n'
               "Content-Type: audio/wav\r\n\r\n").encode()
    body = headers + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = Request("http://127.0.0.1:8083/inference", data=body,
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    begin = time.monotonic()
    with urlopen(request, timeout=25) as reply:
        transcript = json.load(reply)["text"].strip()
    return transcript, time.monotonic() - begin

base = Path("runtime/whisper-turbo-benchmark")
models = {
    "large-v3-turbo-q5_0-resident": base / "ggml-large-v3-turbo-q5_0.bin",
}
with TemporaryDirectory(dir=base, prefix="resident-fixture-") as folder:
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
        alternate = base / "fixtures" / (name + ".wav")
        if alternate.exists():
            inputs.append(("alternate-voice", alternate))
            noise = directory / f"{name}-alternate-noise.wav"
            noise_wave(alternate, noise, 12, index * 100 + 12)
            inputs.append(("alternate-voice-12dB", noise))
        for condition, sample in inputs:
            expected = normalize(case["phrase"])
            versions = list(models) if index % 2 == 0 else list(reversed(models))
            for model in versions:
                text, seconds = decode(sample, directory / "result", models[model])
                edits = distance(expected, normalize(text))
                row = {"case": name, "condition": condition, "model": model,
                       "edits": edits, "chars": len(expected), "seconds": round(seconds, 3),
                       "query": text}
                print(json.dumps(row, ensure_ascii=False), flush=True)
                collected[model].append(row)
    for model, rows in sorted(collected.items()):
        times=sorted(row["seconds"] for row in rows)
        print(json.dumps({"summary": {"model": model, "samples": len(rows),
              "total_edits": sum(row["edits"] for row in rows), "total_chars": sum(row["chars"] for row in rows),
              "exact": sum(row["edits"] == 0 for row in rows),
              "alternate_edits": sum(row["edits"] for row in rows if row["condition"].startswith("alternate")),
              "median_seconds": round(statistics.median(times), 3),
              "p95_seconds": times[math.ceil(.95 * len(times)) - 1]}}), flush=True)
print("WHISPER_RESIDENT_COMPARISON_OK")
