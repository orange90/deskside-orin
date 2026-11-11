import json
from pathlib import Path

print("Orin voice HTTP regression · 10 synthetic Mandarin prompts")
print("Private PulseAudio · loopback only · temporary SQLite")
print()
folder = Path("outputs/orin-desk-agent/evidence")
for stage in ("before", "after"):
    lines = (folder / f"37-voice-suite-{stage}.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines if line.startswith("{")]
    summary = records[-1]["summary"]
    common = summary["common"]["http_p95"]
    general = summary["general"]["http_p95"]
    print(f"{stage:6}  {summary['passed']}/{summary['samples']} passed  ·  P95 common {common:.3f} s  ·  open {general:.3f} s")
    for row in records[:-1]:
        if row["case"] in {"stand", "todo-desk", "general-math"}:
            print(f"  {'PASS' if row['passed'] else 'FAIL':4}  {row['query']} → {row['answer']}")
    print()
