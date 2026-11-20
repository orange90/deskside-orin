from pathlib import Path
import json
import re
import wave
import os
import signal
import subprocess
import tempfile
import threading
import time
from urllib.request import Request, build_opener, ProxyHandler

import store
import model_registry

ROOT = store.ROOT
ASR_HINT = "普通话简体中文。今天坐了多久、站立目标、久坐时长、添加待办、设置提醒。"
NETWORK = build_opener(ProxyHandler({}))


def chat(port, payload, timeout):
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with NETWORK.open(request, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def describe_desk_frames(images, timeout=30, posture_review=False, review_mode=None):
    labels = {
        "standing": "画面显示有人站在桌前。",
        "sitting": "画面显示有人坐在桌前。",
        "away": "画面中未见桌前有人。",
        "unknown": "画面不足，无法判断桌前行为。",
    }
    if posture_review or len(images) == 1:
        prompt = ("Look at the person, not the chairs. Is their torso upright with hips ABOVE "
                  "the chair seat, or are their hips resting ON a seat? Answer only standing, "
                  "sitting, or unknown. The presence of a chair does not mean sitting.")
    else:
        labels.update(standing_up="画面显示桌前的人正在起身。",
                      sitting_down="画面显示桌前的人正在坐下。",
                      leaving="画面显示有人正在离开桌前。")
        prompt = ("The frames are in chronological order. Answer only one visible label: "
                  "standing_up, sitting_down, leaving, standing, sitting, away, unknown.")
    def classify(frames, question, tokens=12):
        raw = chat(8082, {
            "model": "cosmos", "max_tokens": tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                *frames, {"type": "text", "text": question}
            ]}],
        }, timeout)
        return raw.strip().lower().strip("`.*\"' ")

    if review_mode == "person":
        description = classify(images, "Describe the visible human body posture in ONE short sentence, including its spatial relation to furniture. Distinguish standing behind a chair from sitting on it. Do not infer sitting just because a chair overlaps the person. If the person or posture is not visible, say it is unclear.", tokens=32)
        # Accept an explicit assertion about the person. Uncertain descriptions
        # still confirm presence from the detector, without adding posture time.
        match = re.match(r"^(?:the |a )?(?:person|individual|man|woman) is (standing|sitting|seated)\b", description)
        if match and not re.search(r"\b(?:unclear|possibly|maybe|or|might|may|appears|seems)\b", description):
            label = "sitting" if match[1] == "seated" else match[1]
            return label, labels[label]
        return "present", labels["unknown"]
    label = "unknown" if review_mode == "occupancy" else classify(images, prompt)
    if label not in labels or label in {"away", "unknown"}:
        occupancy = classify(images[-1:],
            "Assess whether the desk is occupied. A large chair back can completely hide a seated person. "
            "Do not equate no visible head or legs with nobody present. Answer one label: "
            "present (any human body part visible), occluded (chair or furniture prevents confirming "
            "whether a person is there), empty (work area clearly visible and nobody there).")
        if occupancy == "present":
            return "present", labels["unknown"]
        if occupancy == "occluded":
            return "occluded", "椅背或物体遮挡工位，无法确认人员状态。"
        if occupancy == "empty":
            return "away", labels["away"]
        return "unknown", labels["unknown"]
    return label, labels[label]



def elapsed(seconds):
    hours, remaining = divmod(seconds, 3600)
    minutes, remainder = divmod(remaining, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分钟"
    if minutes:
        return f"{minutes} 分钟"
    return f"{remainder} 秒"


def short_number(token):
    if token.isascii() and token.isdecimal():
        return int(token)
    digits = {char: value for value, char in enumerate("零一二三四五六七八九")}
    digits["两"] = 2
    if token in digits:
        return digits[token]
    if token.count("十") == 1:
        tens, ones = token.split("十")
        if (not tens or tens in digits) and (not ones or ones in digits):
            return 10 * (digits[tens] if tens else 1) + digits.get(ones, 0)
    return None


def explicit_tool(query):
    if todo := re.fullmatch(r"(?:请)?(?:添加|新增)(?:一个|一条)?待办(?:事项)?[：:,，、。\s]*(.+)", query.strip()):
        return {"action": "todo", "body": todo[1]}
    if reminder := re.fullmatch(
        r"(?:请)?([0-9一二两三四五六七八九零十]+)\s*(分钟|小时)后[，,\s]*提醒[，,\s]*我[：:,，\s]*(.+)", query.strip()
    ):
        minutes = short_number(reminder[1])
        if minutes is not None:
            return {"action": "reminder", "body": reminder[3],
                    "minutes": minutes * (60 if reminder[2] == "小时" else 1)}
    return None


def respond(query, request_id, save, voice_reply=False):
    totals = store.snapshot()
    instruction = explicit_tool(query)
    if instruction is None and "坐" in query and any(term in query for term in ("多久", "多长", "时长", "多少")):
        answer = f"今天在桌前坐了 {elapsed(totals['totals'].get('sit', 0))}。"
    elif instruction is None and "站" in query and any(term in query for term in ("多久", "目标", "进度", "多长", "时长", "多少")):
        seconds = totals["totals"].get("stand", 0)
        target = totals["settings"]["stand_minutes"]
        progress = "已达成" if seconds >= target * 60 else "尚未达成"
        answer = f"今天站了 {elapsed(seconds)}，{target} 分钟的目标{progress}。"
    elif instruction is None and any(term in query for term in ("工位", "桌前", "在位")) and any(
        term in query for term in ("待了多久", "在了多久", "在位时长", "待了多长", "在工位多久")
    ):
        answer = f"今天工位检测到人员在位 {elapsed(totals['totals'].get('present', 0))}。"
    else:
        if instruction is None:
            answer_style = ("语音答复直接说结论。简单问题用十个字以内回答。听不清关键词时请用户重说。"
                            if voice_reply else "")
            result = chat(8081, {
                "model": "qwen", "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 160, "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": (
                    "你是本地中文助手。仅返回 JSON。回答问题用 "
                    '{"action":"answer","body":"简短回答"}。添加待办用 '
                    '{"action":"todo","body":"待办内容"}。相对时间提醒用 '
                    '{"action":"reminder","body":"提醒内容","minutes":5}。'
                    "提醒内容只写要做的事，不带时间和提醒我。信息不足时用 answer 询问。" + answer_style)},
                    {"role": "user", "content": query}]
            }, 20)
            instruction = json.loads(result)
        action, body = instruction.get("action"), instruction.get("body")
        if not isinstance(body, str) or not 0 < len(body.strip()) <= 300:
            raise ValueError("模型返回内容无效")
        body = body.strip()
        if action == "answer":
            answer = body
        elif action == "todo":
            body = body.rstrip("。！？!?").strip()
            if not body:
                raise ValueError("待办内容不能为空")
            store.task(request_id, action, body)
            answer = f"已添加待办，{body}。"
        elif action == "reminder":
            body = body.rstrip("。！？!?").strip()
            if not body:
                raise ValueError("提醒内容不能为空")
            minutes = instruction.get("minutes")
            if type(minutes) is not int or not 1 <= minutes <= 1440:
                raise ValueError("请提供 1 到 1440 分钟内的提醒时间")
            store.task(request_id, action, body, minutes)
            answer = f"已设置 {minutes} 分钟后的提醒，{body}。"
        else:
            raise ValueError("模型请求了未知操作")
    if save:
        store.note("dialogue", f"你：{query}\n助手：{answer}")
    return answer


class Voice:
    def __init__(self):
        self.recording = None
        self.lock = threading.RLock()
        self.busy = threading.Event()
        self.env = dict(os.environ)
        self.env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

    def audio_source(self):
        if configured := self.env.get("ORIN_AUDIO_SOURCE"):
            return configured
        sources = subprocess.check_output(["pactl", "list", "short", "sources"],
                                          env=self.env, text=True, timeout=3)
        for line in sources.splitlines():
            fields = line.split()
            if len(fields) > 1:
                name = fields[1]
                if name.startswith("alsa_input.") and re.search(r"Insta360|Mic_Air|_RX", name, re.I):
                    return name
        return subprocess.check_output(["pactl", "get-default-source"], env=self.env,
                                       text=True, timeout=3).strip()

    def start(self):
        with self.lock:
            if self.busy.is_set():
                raise ValueError("语音正在处理中")
            if self.recording:
                raise ValueError("已经在录音")
            temp = tempfile.TemporaryDirectory(prefix="orin-voice-", dir="/dev/shm")
            path = os.path.join(temp.name, "query.pcm")
            command = ["timeout", "--signal=INT", "15s", "parecord", "--raw",
                       "--rate=16000", "--channels=1", "--format=s16le",
                       "--latency-msec=60", "--process-time-msec=20"]
            source = self.audio_source()
            muted = subprocess.run(
                ["pactl", "get-source-mute", source], env=self.env, capture_output=True,
                text=True, timeout=3, check=True
            ).stdout.strip().endswith("yes")
            if muted:
                self.set_mute(source, True, value=0)
            command.extend(["--device", source])
            proc = None
            try:
                proc = subprocess.Popen([*command, path], env=self.env, stderr=subprocess.PIPE,
                                        start_new_session=True)
                capture = Path(path)
                deadline = time.monotonic() + 3
                while not (capture.exists() and capture.stat().st_size >= 640):
                    if proc.poll() is not None or time.monotonic() >= deadline:
                        raise ValueError("麦克风未就绪")
                    time.sleep(.02)
            except (OSError, ValueError):
                try:
                    if proc is not None:
                        if proc.poll() is None:
                            proc.send_signal(signal.SIGINT)
                        try:
                            proc.communicate(timeout=2)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            proc.communicate()
                finally:
                    try:
                        self.set_mute(source, muted, value=1)
                    finally:
                        temp.cleanup()
                raise
            self.recording = (temp, path, proc, source, muted)
            threading.Thread(target=self.expire, args=(proc,), daemon=True).start()

    def set_mute(self, source, change, value):
        if change:
            subprocess.run(["pactl", "set-source-mute", source, str(value)],
                           env=self.env, timeout=3, check=True, capture_output=True)

    def expire(self, proc):
        proc.wait()
        with self.lock:
            if self.recording and self.recording[2] is proc:
                _, _, _, source, muted = self.recording
                try:
                    self.set_mute(source, muted, value=1)
                finally:
                    timer = threading.Timer(15, self.discard, args=(proc,))
                    timer.daemon = True
                    timer.start()

    def discard(self, proc):
        with self.lock:
            if self.recording and self.recording[2] is proc:
                self.recording[0].cleanup()
                self.recording = None

    def stop(self, request_id, save):
        start = time.monotonic()
        with self.lock:
            if not self.recording:
                raise ValueError("没有活动录音")
            temp, path, proc, source, muted = self.recording
            self.recording = None
            self.busy.set()
        try:
            if proc.poll() is None:
                proc.send_signal(2)
            proc.communicate(timeout=5)
            self.set_mute(source, muted, value=1)
            muted = False
            if not os.path.isfile(path) or os.path.getsize(path) < 3200:
                raise ValueError("录音太短或麦克风不可用")
            pcm = Path(path).read_bytes()
            wav_path = Path(path).with_suffix(".wav")
            with wave.open(str(wav_path), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(pcm[:len(pcm) - len(pcm) % 2])
            model = model_registry.asr_path()
            query = self.transcribe(wav_path, model)
            if not query:
                raise ValueError("没有识别到语音")
            answer = respond(query, request_id, save, True)
            seconds = round(time.monotonic() - start, 3)
            return {"query": query, "answer": answer, "answer_ready_seconds": seconds,
                    "seconds": seconds}
        finally:
            try:
                self.set_mute(source, muted, value=1)
            finally:
                temp.cleanup()
                self.busy.clear()

    def transcribe(self, path, model, prompt=ASR_HINT):
        import audioop
        with wave.open(str(path), "rb") as audio:
            if audioop.rms(audio.readframes(audio.getnframes()), audio.getsampwidth()) < 25:
                return ""
        prefix = str(Path(path).with_suffix(".transcript"))
        subprocess.run([
            str(ROOT / "runtime/whisper-cli"), "-m", str(ROOT / "models" / model),
            "-f", path, "-l", "zh", "-t", "4", "--prompt", prompt, "--vad", "-vm",
            str(ROOT / "models/ggml-silero-v6.2.0.bin"), "--vad-speech-pad-ms", "120",
            "-otxt", "-of", prefix
        ], check=True, timeout=25, capture_output=True)
        text = Path(prefix + ".txt").read_text(encoding="utf-8").strip()
        spoken = re.sub(r"[^\w]", "", text)
        hint = re.sub(r"[^\w]", "", prompt)
        if len(spoken) >= 15 and spoken in hint:
            return ""
        return text
