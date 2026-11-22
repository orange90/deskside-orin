from pathlib import Path
import json
import re
import wave
import os
import signal
import subprocess
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import tempfile
import threading
import time
from urllib.request import Request, build_opener, ProxyHandler

import store
import model_registry
import dashboard_data

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


WEEK_PERIOD = r"(?:最近|过去|近)(?:的)?(?:[7七]天|[1一](?:周|(?:个)?星期))"
QUERY_INTRO = (r"(?:请问|请|麻烦)?(?:给我看(?:看)?|帮我(?:看(?:看)?|查(?:询)?)(?:一下)?|"
               r"告诉我|查看|查询|我想知道|我想看看)?(?:我(?:的)?)?")
WEEK_REVIEW = r"(?:工位(?:活动)?|活动|坐站)回顾"
RANGE_HELP = "可以查询单日累计时长，或问「最近七天工位回顾」查看含今天的七天记录。"


def week_answer(metrics, now):
    report = dashboard_data.activity_week(now=now)
    totals = report["totals"]
    days = sum(d["recorded_seconds"] > 0 for d in report["days"])
    presence_days = sum(d["presence_recorded_seconds"] > 0 for d in report["days"])
    header = f"{report['start_date']} 至 {report['end_date']}（含今天），"
    if days:
        header += f"7 天中有 {days} 天留有姿态记录（含未知和隐私时段）。"
    elif presence_days:
        header += f"只有 {presence_days} 天留有工位在位记录，未采集姿态。"
    else:
        return header + "这段时间没有采集到姿态或在位记录。"

    parts = []
    for state, label in (("sit", "坐姿"), ("stand", "站姿")):
        if state in metrics and report["recorded_seconds"]:
            estimated = report["estimated_" + state]
            parts.append(f"{label} {elapsed(totals[state])}（含推算 {elapsed(estimated)}）")
    if "present" in metrics:
        if report["presence_recorded_seconds"]:
            coverage = f"（{presence_days} 天有在位记录）" if days else ""
            parts.append(f"工位在位 {elapsed(totals['present'])}{coverage}")
        else:
            parts.append("未采集到在位记录")
    if "review" in metrics and report["recorded_seconds"]:
        for state, label in (("away", "离开"), ("privacy", "隐私保护"), ("unknown", "姿态未知")):
            parts.append(f"{label} {elapsed(totals[state])}")
    missing = report["elapsed_seconds"] - report["recorded_seconds"]
    if missing and days:
        parts.append(f"未采集姿态 {elapsed(missing)}")
    return header + ("，".join(parts) + "。" if parts else "")


def activity_answer(query, now=None):
    # Match a duration/goal attached to a posture, not a character inside
    # an unrelated noun ("车站") or the object of a reminder.
    if re.search(r"提醒我|待办|帮我记|应该|建议|多久合适", query):
        return None
    duration = r"(?:多久|多长(?:时间)?|(?:多少|几)(?:个)?(?:小时|分钟|秒|时间)|多少(?=[？?。！!\s]*$)|时长|时间(?:是多少|多长))"
    posture_suffix = r"(?:姿(?:势)?|着|立|了|的|累计|总共|一共|总|各|分别|持续)*"
    pair = r"(?:坐(?:姿)?(?:和|与|、|跟|及)?站(?:姿|立)?|站(?:姿|立)?(?:和|与|、|跟|及)?坐(?:姿)?)"
    presence = r"(?:在位|工位|桌前)(?:在位|停留|待|在|了|的|累计|总共|一共|总)*" + duration
    normalized = re.sub(r"\s+", "", query).strip("？?。！!")
    match = re.search(WEEK_PERIOD, normalized)
    if match:
        # Only an explicit request for local activity may consume a week of records.
        intro = normalized[:match.start()]
        if not re.fullmatch(QUERY_INTRO, intro):
            mixed = r"(?:今天|昨天|前天|本周|上周|\d+月\d+[日号])(?:和|与|、|跟)"
            return RANGE_HELP if re.fullmatch(QUERY_INTRO + mixed, intro) else None
        topic = re.sub(r"^(?:里|内)?[，,]?(?:我的|我|的)?", "", normalized[match.end():])
        topic = re.sub(r"(?:呢|吗)$", "", topic)
        if re.search(r"写|翻译|提醒|问卷|采访", topic):
            return None
        if re.fullmatch(WEEK_REVIEW, topic):
            return week_answer({"review", "sit", "stand", "present"}, now)
        topic = re.sub(r"^在(?:桌前|工位)(?=[坐站])", "", topic)
        measures = [(pair + posture_suffix + duration, {"sit", "stand"}),
                    ("坐" + posture_suffix + duration, {"sit"}),
                    ("站" + posture_suffix + duration, {"stand"}), ("(?:在)?" + presence, {"present"})]
        wanted = set()
        for phrase in re.split(r"[，,]|(?<=[久间长钟秒时])[、和]", topic):
            metrics = next((m for pattern, m in measures if re.fullmatch(pattern + r"(?:是多少)?", phrase)), None)
            if metrics is None:
                break
            wanted.update(metrics)
        else:
            return week_answer(wanted, now)
        return RANGE_HELP if re.search(pair + "|" + presence + "|[坐站].*(?:多久|时长)", topic) else None
    if re.fullmatch(QUERY_INTRO + r"(?:本|这|上|下)(?:周|星期)的?" + WEEK_REVIEW, normalized):
        return RANGE_HELP

    combined = re.search(pair + posture_suffix + duration, query)
    # Joining punctuation such as "坐了多久、站了多久" works independently.
    sitting = bool(combined or re.search(r"坐" + posture_suffix + duration, query))
    standing = bool(combined or re.search(r"站" + posture_suffix + duration, query)
                    or re.search(r"站(?:立|姿)?(?:的)?(?:目标|进度)", query))
    present = bool(re.search(presence, query))
    if not (sitting or standing or present):
        return None

    now = now or datetime.now(store.TZ)
    today = now.date()
    requested = set()
    for token, offset in (("今天", 0), ("昨天", 1), ("前天", 2), ("明天", -1)):
        if token in query:
            requested.add(today - timedelta(days=offset))
    for year, month, day in re.findall(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]", query):
        try:
            requested.add(date(int(year or today.year), int(month), int(day)))
        except ValueError:
            return "请给出有效的查询日期。"
    for iso in re.findall(r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b", query, re.ASCII):
        try:
            requested.add(date.fromisoformat(iso))
        except ValueError:
            return "请给出有效的查询日期。"
    restricted_period = any(word in query for word in
            ("本周", "这周", "上周", "本月", "上月", "最近", "近期", "这几天", "过去", "上个月", "本星期",
             "上午", "下午", "早上", "晚上", "本次", "这次", "这一次", "连续", "到现在", "几点")) or re.search(
            r"(?:[0-9几半一二两三四五六七八九十]+(?:点|天|周|(?:个)?星期|个月)|"
            r"(?:这|上|下)(?:个)?(?:星期|礼拜)|[0-9]{1,2}:[0-9]{2})", query)
    if len(requested) > 1 or restricted_period:
        return RANGE_HELP
    day = next(iter(requested), today)
    if day > today:
        return "这一天还没有活动记录。"
    prefix = {today: "今天", today-timedelta(days=1): "昨天",
              today-timedelta(days=2): "前天"}.get(day, day.isoformat())
    start = datetime.combine(day, datetime.min.time(), store.TZ)
    cutoff = min((start+timedelta(days=1)).timestamp(), now.timestamp())
    with store.database() as db:
        db.execute("BEGIN")
        rows, _ = store.activity_window(db, start.timestamp(), cutoff, now.timestamp())
        target = db.execute("SELECT value FROM settings WHERE key='stand_minutes'").fetchone()[0]
    if not any(r['kind'] in {'presence', 'person'} for r in rows):
        return prefix + "没有已记录的活动区间。"
    totals = store.activity_totals(rows)
    answers = []
    for wanted, state, verb in [(sitting, "sit", "在桌前坐了"), (standing, "stand", "站了")]:
        if wanted:
            seconds = totals.get(state, 0)
            estimate = round(sum((r["end"]-r["start"]) for r in rows
                                 if r["kind"] == "person" and r["state"] == state
                                 and r.get("source") in {"estimated", "continuity"}))
            detail = f"，其中 {elapsed(estimate)} 为推算" if estimate else ""
            answers.append(f"{verb} {elapsed(seconds)}{detail}")
    if present:
        answers.append(f"工位检测到人员在位 {elapsed(totals.get('present',0))}")
    if standing and ("目标" in query or "进度" in query):
        progress = "已达成" if totals.get("stand", 0) >= target * 60 else "尚未达成"
        answers.append(f"按当前 {target} 分钟目标，{progress}")
    return prefix + "，".join(answers) + "。"


def respond(query, request_id, save, voice_reply=False):
    if save:
        previous = store.previous_answer(request_id, query)
        if previous is not None:
            return previous
    new_task = None
    instruction = explicit_tool(query)
    answer = activity_answer(query) if instruction is None else None
    if answer is None:
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
            new_task = {"kind": action, "body": body}
            answer = f"已添加待办，{body}。"
        elif action == "reminder":
            body = body.rstrip("。！？!?").strip()
            if not body:
                raise ValueError("提醒内容不能为空")
            minutes = instruction.get("minutes")
            if type(minutes) is not int or not 1 <= minutes <= 1440:
                raise ValueError("请提供 1 到 1440 分钟内的提醒时间")
            new_task = {"kind": action, "body": body, "minutes": minutes}
            answer = f"已设置 {minutes} 分钟后的提醒，{body}。"
        else:
            raise ValueError("模型请求了未知操作")
    return store.record_answer(request_id, query, answer, save, new_task)


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
            command = ["timeout", "--signal=INT", "15s", "parecord", "--raw",
                       "--rate=16000", "--channels=1", "--format=s16le",
                       "--latency-msec=60", "--process-time-msec=20"]
            source = self.audio_source()
            muted = subprocess.run(
                ["pactl", "get-source-mute", source], env=self.env, capture_output=True,
                text=True, timeout=3, check=True
            ).stdout.strip().endswith("yes")
            command.extend(["--device", source])
            temp = tempfile.TemporaryDirectory(prefix="orin-voice-", dir="/dev/shm")
            path = os.path.join(temp.name, "query.pcm")
            proc = None
            try:
                self.set_mute(source, muted, value=0)
                proc = subprocess.Popen([*command, path], env=self.env, stderr=subprocess.PIPE,
                                        start_new_session=True)
                capture = Path(path)
                deadline = time.monotonic() + 3
                while not (capture.exists() and capture.stat().st_size >= 640):
                    if proc.poll() is not None or time.monotonic() >= deadline:
                        raise ValueError("麦克风未就绪")
                    time.sleep(.02)
            except (OSError, ValueError, subprocess.SubprocessError):
                try:
                    if proc is not None:
                        self._stop_capture(proc, timeout=2)
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
                temp = self.recording[0]
                self.recording = None
                temp.cleanup()

    @staticmethod
    def _stop_capture(proc, timeout):
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # parecord belongs to the timeout wrapper's process group.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()

    @contextmanager
    def _transcript(self):
        with self.lock:
            if not self.recording:
                raise ValueError("没有活动录音")
            temp, path, proc, source, muted = self.recording
            self.recording = None
            self.busy.set()
        try:
            self._stop_capture(proc, timeout=5)
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
            if not query or not query.strip():
                raise ValueError("没有识别到语音")
            yield query
        finally:
            try:
                self.set_mute(source, muted, value=1)
            finally:
                try:
                    temp.cleanup()
                finally:
                    self.busy.clear()

    def draft(self):
        start = time.monotonic()
        with self._transcript() as query:
            return {"query": query, "draft": True, "seconds": round(time.monotonic() - start, 3)}

    def stop(self, request_id, save):
        start = time.monotonic()
        with self._transcript() as query:
            answer = respond(query, request_id, save, True)
            seconds = round(time.monotonic() - start, 3)
            return {"query": query, "answer": answer, "answer_ready_seconds": seconds,
                    "seconds": seconds}

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
