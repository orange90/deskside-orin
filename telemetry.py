"""Bounded, read-only hardware telemetry; kept in RAM, never mixed with activity data."""
import os
import logging
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from urllib.request import urlopen


def read_sysfs(path):
    with path.open("rb", buffering=0) as stream:
        raw = stream.read(128)
    if not raw:
        raise ValueError("Sensor unavailable")
    return raw.decode().strip()


class Telemetry:
    def __init__(self):
        self.rows = deque(maxlen=240)  # One hour at 15-second resolution.
        self.lock = threading.Lock()
        self.previous_cpu = None

    def sample(self):
        row = {'at': time.time()}
        try:
            memory = {k: int(v.split()[0]) for k, v in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
            row.update(ram_total_gib=memory['MemTotal']/1024**2,
                       ram_used_gib=(memory['MemTotal']-memory['MemAvailable'])/1024**2,
                       ram_available_gib=memory['MemAvailable']/1024**2,
                       swap_used_gib=(memory['SwapTotal']-memory['SwapFree'])/1024**2,
                       swap_total_gib=memory['SwapTotal']/1024**2)
            values = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
            total, idle = sum(values), values[3] + values[4]
            if self.previous_cpu:
                previous_total, previous_idle = self.previous_cpu
                row['cpu_pct'] = 100*(1-(idle-previous_idle)/max(1, total-previous_total))
            self.previous_cpu = total, idle
            row['uptime_s'] = float(Path('/proc/uptime').read_text().split()[0])
            row['load'] = list(os.getloadavg())
            row['cpu_online'] = Path('/sys/devices/system/cpu/online').read_text().strip()
        except (OSError, ValueError, KeyError):
            pass
        disk = shutil.disk_usage(Path(__file__).parent)
        row.update(disk_total_gib=disk.total/1024**3, disk_free_gib=disk.free/1024**3,
                   disk_used_gib=disk.used/1024**3)
        temperatures = {}
        for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
            try:
                temperatures[read_sysfs(path.parent/'type')] = int(read_sysfs(path)) / 1000
            except (OSError, ValueError):
                pass
        row['temperatures'] = temperatures
        if temperatures:
            row['temp_c'] = max(temperatures.values())
        for path in Path('/sys/class/devfreq').glob('*.gpu/cur_freq'):
            try:
                row['gpu_mhz'] = int(read_sysfs(path)) / 1e6
            except (OSError, ValueError):
                pass
        row['fan_rpm'] = {}
        for path in Path('/sys/class/hwmon').glob('hwmon*/rpm'):
            try:
                row['fan_rpm'][path.parent.name] = int(read_sysfs(path))
            except (OSError, ValueError):
                pass
        executable = shutil.which('tegrastats')
        if executable:
            process = None
            try:
                process = subprocess.Popen([executable, '--interval', '1000'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                try:
                    output, _ = process.communicate(timeout=1.3)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    output, _ = process.communicate(timeout=1)
                line = output.strip().splitlines()[-1] if output.strip() else ''
                row.update(parse_tegrastats(line))
            except (OSError, subprocess.TimeoutExpired):
                if process and process.poll() is None:
                    process.kill()
                    process.communicate()
        services = {}
        for name, port in [('Qwen', 8081), ('Cosmos', 8082)]:
            try:
                with urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                    services[name] = response.status == 200
            except OSError:
                services[name] = False
        row['services'] = services
        return row

    def run(self, stop):
        while not stop.is_set():
            started = time.monotonic()
            try:
                row = self.sample()
                with self.lock:
                    self.rows.append(row)
            except Exception as error:
                logging.getLogger("uvicorn.error").warning("Hardware sample failed: %s", type(error).__name__)
            stop.wait(max(0, 15-(time.monotonic()-started)))

    def snapshot(self):
        with self.lock:
            rows = list(self.rows)
        return {'latest': rows[-1] if rows else None, 'history': rows,
                'interval_seconds': 15, 'retention_seconds': 3600}


def parse_tegrastats(line):
    result = {'power_rails_w': {}}
    match = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if match:
        result['gpu_pct'] = int(match.group(1))
    for name, instant, average in re.findall(r'\b((?:VDD|VIN)_[A-Z0-9_]+)\s+(\d+)mW/(\d+)mW', line):
        result['power_rails_w'][name] = int(instant)/1000
    match = re.search(r'CPU\s+\[([^]]+)\]', line)
    if match:
        result['cpu_cores'] = [None if x == 'off' else int(x.split('%')[0]) for x in match.group(1).split(',')]
    return result


monitor = Telemetry()
