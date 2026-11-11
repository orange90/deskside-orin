"""Compare local ASR engines on one short recording per prompt."""
import json
from pathlib import Path
from runpy import run_path
import time
from uuid import uuid4
import wave

import numpy as np
import sherpa_onnx

import assistant

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(__file__).with_name("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=str(MODEL / "model.int8.onnx"), tokens=str(MODEL / "tokens.txt"),
    language="zh", use_itn=True, num_threads=2, provider="cpu",
)

class ComparingVoice(assistant.Voice):
    def transcribe(self, path, model):
        start = time.monotonic()
        whisper = super().transcribe(path, model)
        whisper_time = time.monotonic() - start
        start = time.monotonic()
        with wave.open(str(path), "rb") as audio:
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        recognizer.decode_stream(stream)
        sensevoice = stream.result.text
        self.results = (whisper, sensevoice, whisper_time, time.monotonic() - start)
        return sensevoice or whisper

assistant.respond = lambda *args, **kwargs: ""
voice = ComparingVoice()
cases = run_path(str(ROOT / "voice-field-check.py"))["suite_cases"](10)
path = ROOT / "logs" / f"field-asr-compare-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
with path.open("x") as log:
    for index in (3, 7, 9, 10):
        _, phrase, noisy = cases[index - 1]
        print(f"原第 {index} 条 · {'轻微背景噪声' if noisy else '安静'}\n请说：{phrase}")
        input("按回车开始录音。")
        voice.start()
        try:
            input("说完按回车结束录音。")
        finally:
            try:
                voice.stop(str(uuid4()), False)
            except ValueError:
                pass
        w, s, wt, st = voice.results
        print(f"Whisper：{w or '无结果'}\nSenseVoice：{s or '无结果'}")
        choice = input("哪条识别正确？w = Whisper，s = SenseVoice，b = 都正确，直接回车 = 都不正确。").strip().lower()
        row = {"index": index, "noise": noisy, "whisper_ok": choice in {"w", "b"},
               "sensevoice_ok": choice in {"s", "b"}, "whisper_seconds": round(wt, 3),
               "sensevoice_seconds": round(st, 3)}
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
print(f"核对结果保存在 Orin {path}。")
