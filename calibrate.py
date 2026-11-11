"""Interactive desk calibration using metadata only. Runs on the Orin."""
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8765"

steps = (
    ("low", "把桌面降到常用坐姿档，等桌面停稳后按回车"),
    ("high", "把桌面升到常用站姿档，等桌面停稳后按回车"),
)
print("本命令仅校准桌面档位。先在看板选中桌沿的移动范围。")
for key, prompt in steps:
    while True:
        input(prompt + "。")
        try:
            request = Request(f"{BASE}/api/calibration/{key}", data=b"{}",
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=3) as response:
                print("已记录", json.load(response))
            break
        except HTTPError as error:
            print("请调整后重试。", json.load(error)["detail"])
print("桌面高低两个参考点已记录。请再升降桌三次，最后把桌面留在高档并离开桌前。")
