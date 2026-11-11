import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()
module_spec = importlib.util.spec_from_file_location("store_under_test", args.source)
store = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(store)

kinds = {"person": ("sit", "stand"), "presence": ("present", "present"), "desk": ("low", "high")}
start = datetime.now(store.TZ).replace(hour=10, minute=0, second=0, microsecond=0).timestamp()
with TemporaryDirectory(dir="/dev/shm", prefix="orin-gap-") as temp:
    store.ROOT = Path(temp)
    store.initialize()
    for kind, (first, _) in kinds.items():
        store.persist(kind, first, start, False)
        store.persist(kind, first, start + 10, True)
    for second in range(20, 1820, 2):
        for kind in kinds:
            store.persist(kind, "unknown", start + second, False)
    for kind, (_, restored) in kinds.items():
        store.persist(kind, restored, start + 1820, False)
        store.persist(kind, restored, start + 1830, True)

    with store.database() as db:
        for kind, (first, restored) in kinds.items():
            rows = db.execute("SELECT state,start,end FROM intervals WHERE kind=? ORDER BY start", (kind,)).fetchall()
            spans = [(row["state"], row["start"] - start, row["end"] - start) for row in rows]
            unknown_rows = [(a, b) for state, a, b in spans if state == "unknown"]
            assert spans[0] == (first, 0, 10) and spans[-1] == (restored, 1820, 1830), spans[:4]
            unknown_seconds = sum(b - a for a, b in unknown_rows)
            assert abs(unknown_seconds - 1810) < .002
            if not args.baseline:
                assert len(spans) == 3 and unknown_rows == [(10, 1820)], spans
            print(json.dumps({"kind": kind, "rows": len(spans), "unknown_seconds": round(unknown_seconds, 6)}), flush=True)
        pages = db.execute("PRAGMA page_count").fetchone()[0]
        print(f"SQLITE_PAGES={pages}", flush=True)

    totals = store.snapshot()["totals"]
    assert totals["sit"] == totals["stand"] == 10 and totals["present"] == 20
    store.persist("restart-case", "sit", start, False)
    store.persist("restart-case", "sit", start + 10, True)
    store.persist("restart-case", "sit", start + 130, False)
    with store.database() as db:
        rows = db.execute("SELECT state,end-start FROM intervals WHERE kind='restart-case' ORDER BY start").fetchall()
        assert [tuple(row) for row in rows] == [("sit", 10), ("unknown", 120), ("sit", 0)]
    print("OUTAGE_TOTALS_AND_RESTART_GAP_OK")
