from array import array
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import wave

cases = json.loads(Path("cases.json").read_text())
Path("fixtures").mkdir(exist_ok=True)
for index, case in enumerate(cases):
    path = Path("fixtures") / (case["id"] + ".wav")
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "source.aiff"
        subprocess.run(["say", "-v", "Tingting", "-o", str(source), case["phrase"]], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(source), str(path)], check=True)
    if case.get("noise"):
        with wave.open(str(path), "rb") as audio:
            params = audio.getparams()
            values = array("h", audio.readframes(audio.getnframes()))
        rms = math.sqrt(sum(value * value for value in values) / len(values))
        noise = random.Random(index)
        for i, value in enumerate(values):
            values[i] = max(-32768, min(32767, round(value + noise.gauss(0, rms * .1))))
        with wave.open(str(path), "wb") as audio:
            audio.setparams(params)
            audio.writeframes(values.tobytes())
    assert path.stat().st_size > 3200
print(f"Created {len(cases)} synthetic WAV files.")
