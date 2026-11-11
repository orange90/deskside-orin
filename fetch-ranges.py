"""Download public model bytes from their resolved CDN URLs, with resumable chunks."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent / "models"
CHUNK = 64 * 1024 * 1024


def part(url, path, start, end, size):
    if path.is_file() and path.stat().st_size == end - start + 1:
        return
    tmp = path.with_suffix(".tmp")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=45) as source, tmp.open("wb") as file:
                if source.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise RuntimeError("CDN range mismatch")
                shutil.copyfileobj(source, file, 1024 * 1024)
            if tmp.stat().st_size != end - start + 1:
                raise RuntimeError("incomplete range")
            tmp.replace(path)
            return
        except Exception as error:
            print(f"Range {start}-{end}, attempt {attempt+1}: {type(error).__name__}", flush=True)
            if attempt == 3:
                raise RuntimeError("range failed") from None
            time.sleep(2 ** attempt)


def fetch(model):
    name, url, size, digest = model
    dest = ROOT / name
    if dest.exists() and dest.stat().st_size == size:
        print(f"Already present {name}", flush=True)
        return
    chunks = dest.with_suffix(dest.suffix + ".chunks")
    chunks.mkdir(parents=True, exist_ok=True)
    pieces = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = []
        for start in range(0, size, CHUNK):
            end = min(start + CHUNK, size) - 1
            piece = chunks / f"{start:014d}"
            pieces.append(piece)
            jobs.append(pool.submit(part, url, piece, start, end, size))
        for count, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            job.result()
            print(f"{name}: {count}/{len(jobs)} chunks", flush=True)
    joined = dest.with_suffix(dest.suffix + ".joined")
    hasher = hashlib.sha256()
    with joined.open("wb") as file:
        for piece in pieces:
            with piece.open("rb") as segment:
                while block := segment.read(1024 * 1024):
                    file.write(block)
                    hasher.update(block)
    if hasher.hexdigest() != digest:
        raise RuntimeError(f"SHA256 mismatch for {name}")
    joined.replace(dest)
    shutil.rmtree(chunks)
    print(f"Verified {name} · {size} bytes", flush=True)


if __name__ == "__main__":
    input_path = Path(sys.argv[1])
    models = json.loads(input_path.read_text())
    input_path.unlink()
    for model in models:
        fetch(model)
