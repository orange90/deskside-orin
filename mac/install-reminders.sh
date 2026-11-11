#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
app="$HOME/Applications/桌间提醒.app"
support="$HOME/Library/Application Support/OrinDeskAgent"
label=local.orin.desk-agent.reminders
agent="$HOME/Library/LaunchAgents/$label.plist"
mkdir -p "$app/Contents/MacOS" "$support" "$HOME/Library/LaunchAgents"
/usr/bin/clang -fobjc-arc DeskNotice.m -o "$app/Contents/MacOS/DeskNotice" -framework AppKit -framework UserNotifications
/usr/bin/python3 - "$app" "$agent" "$support" "$label" <<'PY'
from pathlib import Path
import plistlib,sys
app,agent,support,label=sys.argv[1:]
Path(app,'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'local.orin.deskside.reminders','CFBundleName':'桌间提醒','CFBundleDisplayName':'桌间提醒','CFBundleExecutable':'DeskNotice','CFBundlePackageType':'APPL','CFBundleVersion':'1','LSUIElement':True,'LSMinimumSystemVersion':'12.0'}))
Path(agent).write_bytes(plistlib.dumps({'Label':label,'ProgramArguments':['/usr/bin/python3',str(Path(support,'orin-reminders.py'))],'RunAtLoad':True,'StartInterval':15,'ProcessType':'Background','StandardOutPath':str(Path(support,'reminders.log')),'StandardErrorPath':str(Path(support,'reminders-error.log'))}))
PY
/usr/bin/codesign --force --sign - "$app"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister "$app"
install -m 700 orin-reminders.py "$support/orin-reminders.py"
if [[ ! -f "$support/reminders.cursor" ]]; then /usr/bin/python3 "$support/orin-reminders.py" --initialize; fi
domain="gui/$(id -u)"
if /bin/launchctl print "$domain/$label" >/dev/null 2>&1; then /bin/launchctl bootout "$domain/$label"; fi
/bin/launchctl bootstrap "$domain" "$agent"
/usr/bin/plutil -lint "$agent"
"$app/Contents/MacOS/DeskNotice" --status
