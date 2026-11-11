"""Run the local speech-to-text and text reply path on synthetic WAV fixtures."""
import json
from pathlib import Path
import tempfile
import time
import wave

from assistant import Voice
import store

ROOT = Path(__file__).resolve().parent


class FinishedRecording:
    def poll(self):
        return 0

    def communicate(self, timeout):
        return b"", b""


def main():
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="orin-voice-smoke-") as fixture_db:
        store.ROOT = Path(fixture_db)
        store.initialize()
        now = time.time()
        store.persist("person", "sit", now - 120, False)
        store.persist("person", "sit", now, True)
        for case, expected in (("stat", "2 分钟"), ("reminder", "喝水")):
            voice = Voice()
            recording = tempfile.TemporaryDirectory(dir="/dev/shm", prefix="orin-voice-input-")
            path = Path(recording.name) / "query.pcm"
            with wave.open(str(ROOT / "fixtures" / f"voice-{case}.wav"), "rb") as audio:
                assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
                path.write_bytes(audio.readframes(audio.getnframes()))
            voice.recording = (recording, str(path), FinishedRecording(), "", False)
            reply = voice.stop(f"synthetic-{case}", False)
            assert expected in reply["answer"], reply
            print(json.dumps({"case": case, **reply}, ensure_ascii=False), flush=True)
        with store.database() as db:
            task = db.execute("SELECT kind,body,due FROM tasks WHERE id='synthetic-reminder'").fetchone()
            assert task["kind"] == "reminder" and task["body"] == "喝水"
            assert 290 <= task["due"] - time.time() <= 300
            db.execute("UPDATE tasks SET due=? WHERE id='synthetic-reminder'", (time.time() - 1,))
        store.record_due_reminders()
        store.record_due_reminders()
        assert [(r["kind"], r["body"]) for r in store.snapshot()["notes"]] == [("reminder", "喝水")]
        print("Due reminder recorded once in the dashboard feed.")


if __name__ == "__main__":
    main()
