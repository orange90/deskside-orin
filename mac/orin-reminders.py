#!/usr/bin/env python3
"""Deliver local reminder metadata to this Mac via SSH; no audio, images or cloud."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
SUPPORT = Path.home()/'Library/Application Support/OrinDeskAgent'
CURSOR = SUPPORT/'reminders.cursor'
HELPER = Path.home()/'Applications/桌间提醒.app/Contents/MacOS/DeskNotice'


def feed(cursor):
    command = f"wget --no-proxy -qO- --timeout=5 --tries=1 'http://127.0.0.1:8765/api/reminders?after={int(cursor)}'"
    result = subprocess.run(['/usr/bin/ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','orin',command],capture_output=True,text=True,timeout=12,check=True)
    return json.loads(result.stdout)


def save_cursor(value):
    temporary=CURSOR.with_suffix('.tmp');temporary.write_text(str(value)+'\n');temporary.replace(CURSOR)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--initialize',action='store_true');parser.add_argument('--test',action='store_true');args=parser.parse_args()
    SUPPORT.mkdir(parents=True,exist_ok=True,mode=0o700)
    if args.test:
        return subprocess.run([str(HELPER),'--notify','桌间 · 通知测试','已连接 Orin。本机久坐和定时提醒会在这里显示，无声音。','deskside-connection-test'],timeout=90).returncode
    cursor=int(CURSOR.read_text()) if CURSOR.exists() else None
    response=feed(cursor or 0)
    if args.check:
        print(json.dumps({'connected':True,'cursor':cursor,'latest_id':response['latest_id'],'pending':len(response['reminders'])},ensure_ascii=False));return 0
    if args.initialize or cursor is None:
        save_cursor(response['latest_id']);print('Initialized at',response['latest_id']);return 0
    for reminder in response['reminders']:
        # Reconnect without a flood: omit reminders older than 12 hours, keeping them on Orin.
        if time.time()-reminder['at'] <= 43200:
            when=datetime.fromtimestamp(reminder['at']).astimezone().strftime('%H:%M')
            result=subprocess.run([str(HELPER),'--notify','桌间 · Orin 提醒',f"{when} · {reminder['body']}",f"orin-reminder-{reminder['id']}"],capture_output=True,text=True,timeout=20)
            if result.returncode:
                raise RuntimeError(result.stdout.strip() or 'Notification submission failed')
            print(datetime.now().isoformat(timespec='seconds'),'submitted reminder',reminder['id'],flush=True)
        save_cursor(reminder['id'])
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        print(datetime.now().isoformat(timespec='seconds'),type(error).__name__,str(error),file=sys.stderr);sys.exit(1)
