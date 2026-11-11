import json
import re
import subprocess
import time
import unicodedata
from urllib.request import Request, urlopen

styles = {
    "current": "语音答复用一句中文，尽量不超过16个字。",
    "concise": "语音答复直接说结论。简单问题用十个字以内回答。听不清关键词时请用户重说。",
    "literal": "简单问题直接答结论，用十个字以内。依据语音转写原意，关键词含义不清时用短句追问。",
}
system = (
    '你是本地中文助手。仅返回 JSON。回答问题用 {"action":"answer","body":"简短回答"}。'
    '添加待办用 {"action":"todo","body":"待办内容"}。相对时间提醒用 '
    '{"action":"reminder","body":"提醒内容","minutes":5}。'
    '提醒内容只写要做的事，不带时间和提醒我。信息不足时用 answer 询问。'
)
terminal = subprocess.check_output(
    ["tmux", "capture-pane", "-p", "-S", "-", "-J", "-t", "voice-field-10:voice"], text=True)
section = re.search(r"第 10/10 条[^\n]*\n(.*?)(?=验收结果|\Z)", terminal, re.S)[1]
actual = re.search(r"识别到[：:]\s*([^\n]+)", section)[1]
actual = "".join(c for c in actual if unicodedata.category(c)[0] in "LN" or c in "。？！?,，")
assert "几条边" in actual and "三角形" not in actual

cases = (
    ("sky", "为什么天空通常是蓝色的？", "answer", r"散射|波长"),
    ("math", "七乘八等于多少？", "answer", r"56|五十六"),
    ("fraction", "七成八等于多少？", "answer", r"78|七十八|0[.]78|零点七八|请确认"),
    ("discount", "打七成八相当于百分之多少？", "answer", r"78|七十八|0[.]78|零点七八"),
    ("geometry", "三角形有几条边？", "answer", r"3\s*条|三条"),
    ("field-misheard", actual, "answer", r"重说|再说|听清|确认|指的|哪种|什么形"),
    ("general-complex", "写一句中文，解释闰年多出一天的原因。", "answer", r"公转|回归年|太阳|365|地球|历法"),
    ("natural-todo", "把明天检查台灯这件事加入我的待办。", "todo", r"检查台灯"),
    ("natural-reminder", "我想五分钟以后收到喝水的提醒。", "reminder", r"喝水"),
)
for name, style in styles.items():
    for case, query, action, expected in cases:
        payload = {"model": "qwen", "max_tokens": 160, "temperature": .1, "seed": 42,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system + style},
                                {"role": "user", "content": query}]}
        request = Request("http://127.0.0.1:8081/v1/chat/completions",
                          data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        start = time.monotonic()
        with urlopen(request, timeout=20) as reply:
            response = json.load(reply)
        duration = time.monotonic() - start
        body = json.loads(response["choices"][0]["message"]["content"])
        passed = body.get("action") == action and bool(re.search(expected, body.get("body", "")))
        if action == "reminder":
            passed &= body.get("minutes") == 5
        row = {"style": name, "case": case, "seconds": round(duration, 3),
               "tokens": response["usage"]["completion_tokens"], "passed": passed}
        if case != "field-misheard":
            row["reply"] = body.get("body")
        print(json.dumps(row, ensure_ascii=False), flush=True)
