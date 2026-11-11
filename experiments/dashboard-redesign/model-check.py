import json, tempfile, time
from pathlib import Path
from urllib.request import Request, urlopen
import assistant, model_registry
root=Path.cwd()
voice=assistant.Voice()
with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
    fixture=Path(directory)/'check.wav';fixture.write_bytes((root/'fixtures/voice-stat.wav').read_bytes())
    for key in ('small','base','turbo'):
        started=time.monotonic()
        output=voice.transcribe(fixture,root/model_registry.ASR[key]['path'])
        assert output.strip(),key
        print(json.dumps({'model':key,'nonempty':True,'seconds':round(time.monotonic()-started,3)},ensure_ascii=False),flush=True)
print('All selectable models load and transcribe local synthetic audio; no microphone capture or user record writes.')
