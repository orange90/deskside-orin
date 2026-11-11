#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/orin-desk-agent"
backup="backups/dashboard-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup"
cp main.py assistant.py dashboard.html state.py vision.py store.py "$backup/"
python3 - "$backup" <<'PY'
import json,sqlite3,sys,urllib.request
from pathlib import Path
with urllib.request.urlopen('http://127.0.0.1:8765/api/state',timeout=5) as response:
    assert not json.load(response)['recording'], 'Recording active; deployment deferred'
with sqlite3.connect('data.sqlite3') as source, sqlite3.connect(Path(sys.argv[1])/'data.sqlite3') as target:
    source.backup(target)
    print('BACKUP',sys.argv[1],{name:source.execute('SELECT COUNT(*) FROM '+name).fetchone()[0] for name in ['intervals','notes','tasks','settings']},flush=True)
Path('runtime/dashboard-redesign/backup-path').write_text(sys.argv[1])
PY
for file in dashboard.html dashboard_data.py telemetry.py model_registry.py assistant.py main.py state.py vision.py store.py activity_annotations.py; do
  cp "runtime/dashboard-redesign/$file" "$file.new"
  mv "$file.new" "$file"
done
tmux send-keys -t '=orin-agent:=vision' C-c
for n in $(seq 1 20); do
  if ! tmux list-windows -t '=orin-agent' -F '#{window_name}' | grep -Fxq vision; then break; fi
  sleep 1
done
bash stack.sh start
python3 - <<'PY'
import json,time,urllib.request,sqlite3
from pathlib import Path
for attempt in range(90):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/api/dashboard',timeout=3) as response:
            data=json.load(response)
        if data['camera']['ready']: break
    except Exception:pass
    time.sleep(1)
else:raise SystemExit('Dashboard health check failed')
backup=Path('runtime/dashboard-redesign/backup-path').read_text().strip()
with sqlite3.connect('data.sqlite3') as db:
    db.execute('ATTACH DATABASE ? AS old',(str(Path(backup)/'data.sqlite3'),))
    assert db.execute('SELECT COUNT(*) FROM old.notes o LEFT JOIN notes n ON n.id=o.id WHERE n.id IS NULL OR n.body!=o.body OR n.kind!=o.kind OR n.at!=o.at').fetchone()[0]==0
    assert db.execute('SELECT COUNT(*) FROM old.intervals o LEFT JOIN intervals n ON n.kind=o.kind AND n.start=o.start WHERE n.start IS NULL OR n.state!=o.state OR n.end<o.end').fetchone()[0]==0
    assert db.execute('SELECT COUNT(*) FROM old.tasks o LEFT JOIN tasks n ON n.id=o.id WHERE n.id IS NULL OR n.body!=o.body OR n.kind!=o.kind OR n.due IS NOT o.due').fetchone()[0]==0
    assert db.execute('SELECT COUNT(*) FROM old.settings o LEFT JOIN settings n ON n.key=o.key WHERE n.key IS NULL OR n.value!=o.value').fetchone()[0]==0
print('DEPLOY PASS; all original notes, intervals, tasks and settings preserved',flush=True)
print('Today totals',data['totals']);print('Notes',data['raw_note_count'],'->',len(data['notes']),'groups')
PY
bash stack.sh status
