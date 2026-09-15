import json
import sqlite3
import threading
import time
import uuid

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class JobStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL, body TEXT)")
        self.db.commit()

    def get(self, job_id):
        with self.lock:
            row = self.db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def all(self):
        with self.lock:
            return [json.loads(row[0]) for row in self.db.execute("SELECT body FROM jobs ORDER BY created")]

    def update(self, job_id, **fields):
        with self.lock:
            item = self.get(job_id)
            if item is None:
                raise KeyError(job_id)
            item.update(fields)
            self.db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(item), job_id))
            self.db.commit()
            return item

    def add(self, request, max_queue):
        with self.lock:
            count = sum(job["status"] not in TERMINAL for job in self.all())
            if count >= max_queue:
                raise OverflowError("Job queue is full")
            item = {"id": uuid.uuid4().hex, "created_at": time.time(), "status": "queued", "request": request}
            self.db.execute(
                "INSERT INTO jobs VALUES (?,?,?)", (item["id"], item["created_at"], json.dumps(item))
            )
            self.db.commit()
            return item

    def recover(self):
        for job in self.all():
            if job["status"] in {"loading", "running"}:
                self.update(
                    job["id"], status="interrupted", finished_at=time.time(), error="Controller restarted"
                )

    def cancel(self, job_id):
        with self.lock:
            item = self.get(job_id)
            if item is None:
                raise KeyError(job_id)
            if item["status"] == "queued":
                return self.update(job_id, status="cancelled", finished_at=time.time())
            if item["status"] not in TERMINAL:
                return self.update(job_id, cancel_requested=True)
            return item

    def claim(self):
        with self.lock:
            for job in self.all():
                if job["status"] == "queued":
                    return self.update(job["id"], status="loading", started_at=time.time())
        return None

    def close(self):
        self.db.close()
