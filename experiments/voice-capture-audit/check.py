from array import array
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
import wave

import assistant

assistant.respond = lambda *args, **kwargs: ""
rate = 16000
length = .8

class CaptureAudit(assistant.Voice):
    samples = None

    def transcribe(self, path, model):
        with wave.open(str(path), "rb") as audio:
            self.samples = array("h", audio.readframes(audio.getnframes()))
        return "fixture"

def segment_duration(samples, frequency):
    width = 320
    oscillator_sin = [math.sin(2*math.pi*frequency*n/rate) for n in range(width)]
    oscillator_cos = [math.cos(2*math.pi*frequency*n/rate) for n in range(width)]
    duration = 0
    for offset in range(0, len(samples)-width, width):
        window = samples[offset:offset+width]
        sine = sum(sample*weight for sample, weight in zip(window, oscillator_sin))
        cosine = sum(sample*weight for sample, weight in zip(window, oscillator_cos))
        if 2*math.hypot(sine, cosine)/width > 1600:
            duration += width/rate
    return round(duration, 3)

with TemporaryDirectory(dir="/dev/shm", prefix="orin-capture-audit-") as temporary:
    fixture = Path(temporary) / "pilot.wav"
    data = array("h")
    for i in range(round(length*rate)):
        frequency = 700 if i < rate*.16 else 1600 if i >= rate*.64 else 1100
        data.append(round(8000*math.sin(2*math.pi*frequency*i/rate)))
    with wave.open(str(fixture), "wb") as file:
        file.setparams((1,2,rate,0,"NONE","not compressed"))
        file.writeframes(data.tobytes())
    for index in range(5):
        voice = CaptureAudit()
        start = time.monotonic()
        voice.start()
        returned = time.monotonic()
        pcm = Path(voice.recording[1])
        bytes_ready = pcm.stat().st_size if pcm.exists() else 0
        subprocess.run(["paplay","--device","orin_fixture",str(fixture)],check=True,timeout=5)
        voice.stop(f"capture-{index}",False)
        print(json.dumps({"start_seconds":round(returned-start,3),"pcm_bytes_at_start_return":bytes_ready,
                          "first_tone_seconds":segment_duration(voice.samples,700),
                          "last_tone_seconds":segment_duration(voice.samples,1600)}), flush=True)
