"""Compare local ASR engines on one short recording per prompt."""
import json
from pathlib import Path
from runpy import run_path
import time
from uuid import uuid4
import assistant

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(__file__).with_name("ggml-large-v3-turbo-q5_0.bin")

class ComparingVoice(assistant.Voice):
    results = None

    def transcribe(self, path, model):
        start = time.monotonic()
        small = super().transcribe(path, "ggml-small.bin")
        small_seconds = time.monotonic() - start
        start = time.monotonic()
        turbo = super().transcribe(path, MODEL)
        self.results = (small, turbo, small_seconds, time.monotonic() - start)
        return turbo or small

assistant.respond = lambda *args, **kwargs: ""
voice = ComparingVoice()
cases = run_path(str(ROOT / "voice-field-check.py"))["suite_cases"](10)
path = ROOT / "logs" / f"field-asr-compare-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
with path.open("x") as log:
    for index in (3, 7, 9, 10):
        _, phrase, noisy = cases[index - 1]
        print(f"原第 {index} 条 · {'轻微背景噪声' if noisy else '安静'}\n请说：{phrase}")
        input("按回车开始录音。")
        voice.results = None
        voice.start()
        try:
            input("说完按回车结束录音。")
        finally:
            try:
                voice.stop(str(uuid4()), False)
            except ValueError:
                pass
        if voice.results is None:
            print("本次未取得可用音频。")
        w, s, wt, st = voice.results or ("", "", 0, 0)
        print(f"Whisper small：{w or '无结果'}\nWhisper Turbo：{s or '无结果'}")
        choice = input("哪条识别正确？w = small，s = Turbo，b = 都正确，直接回车 = 都不正确。").strip().lower()
        row = {"index": index, "noise": noisy, "small_ok": choice in {"w", "b"},
               "turbo_ok": choice in {"s", "b"}, "small_seconds": round(wt, 3),
               "turbo_seconds": round(st, 3)}
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
print(f"核对结果保存在 Orin {path}。")
