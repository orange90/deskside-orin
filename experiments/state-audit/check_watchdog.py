"""Exercise a wedged camera in a separate loopback app with a temporary DB."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import urlopen


def serve(fd, database_dir):
    import threading
    import uvicorn
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import main
    import store

    store.ROOT = Path(database_dir)
    def stalled_capture():
        main.publish(b'fixture', ready=True, person='stand', presence='present',
                     frame_clock=time.monotonic(), captured_at=time.time())
        # Simulate a driver that cannot observe the application's stop flag.
        threading.Event().wait(120)

    main.capture_frames = stalled_capture
    main.monitor.run = lambda stop: stop.wait()
    with socket.socket(fileno=fd) as listener:
        server = uvicorn.Server(uvicorn.Config(main.app, log_level='warning'))
        server.run(sockets=[listener])


def check(folder):
    with socket.socket() as listener, (Path(folder) / 'service.log').open('w+') as log:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        def get(path):
            with urlopen(f'http://127.0.0.1:{port}{path}', timeout=2) as response:
                return response.read()

        proc = subprocess.Popen([sys.executable, __file__, str(listener.fileno()), folder],
                                pass_fds=(listener.fileno(),), stdout=log, stderr=log)
        try:
            start = time.monotonic()
            while True:
                try:
                    if json.loads(get('/api/state'))['camera'].get('ready'):
                        break
                except OSError:
                    pass
                if proc.poll() is not None or time.monotonic() - start > 10:
                    raise RuntimeError('Fixture did not publish a fresh frame')
                time.sleep(.1)
            time.sleep(3.2)
            cam = json.loads(get('/api/state'))['camera']
            assert cam.get('stale') and cam['presence'] == cam['person'] == 'unknown', cam
            for route, status in [('/api/live/frame', 423), ('/api/calibration/frame', 503)]:
                try:
                    get(route)
                except HTTPError as error:
                    assert error.code == status
                else:
                    raise AssertionError('Stalled preview served by ' + route)
            proc.wait(timeout=42)
            log.seek(0)
            assert 'No capture heartbeat' in log.read()
            print(json.dumps(dict(fresh_then_stale=True, previews_blocked=True,
                                  child_exit=proc.returncode, elapsed_s=round(time.monotonic()-start,1))))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        serve(int(sys.argv[1]), sys.argv[2])
    else:
        with tempfile.TemporaryDirectory() as temp:
            check(temp)
