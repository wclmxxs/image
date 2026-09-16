import logging
import threading
import time

from image_lab.common import safe_error
from image_lab.runtime import Cancelled


class Scheduler:
    def __init__(self, settings, registry, store, runtime):
        self.settings, self.registry, self.store, self.runtime = settings, registry, store, runtime
        self.shutdown = threading.Event()
        self.thread = threading.Thread(target=self.run, name="model-scheduler", daemon=True)
        self.current_job = None

    def start(self):
        self.runtime.recover()
        self.store.recover()
        self.thread.start()

    def run(self):
        while not self.shutdown.is_set():
            job = self.store.claim()
            if job is None:
                self.shutdown.wait(self.settings.poll_interval)
                continue
            self.current_job = job["id"]
            began = time.monotonic()

            def interrupted():
                return self.shutdown.is_set() or self.store.get(job["id"]).get("cancel_requested", False)

            try:
                model = self.registry.resolve(job["request"]["model"])
                loading = self.runtime.ensure(model, interrupted)
                self.store.update(job["id"], status="running", load=loading)
                result = self.runtime.generate(job, interrupted)
                if interrupted():
                    raise Cancelled("Job cancelled")
                service_seconds = time.monotonic() - began
                queue_seconds = job["started_at"] - job["created_at"]
                if "timings" in result:
                    result["timings"]["request"] = {
                        "queue_seconds": queue_seconds,
                        "load_seconds": loading["load_seconds"],
                        "service_seconds": service_seconds,
                    }
                self.store.update(
                    job["id"],
                    status="succeeded",
                    result=result,
                    finished_at=time.time(),
                    queue_seconds=queue_seconds,
                    service_seconds=service_seconds,
                    image_url=f"/v1/jobs/{job['id']}/image",
                )
            except Exception as error:
                cancelled = isinstance(error, Cancelled)
                state = "interrupted" if self.shutdown.is_set() else "cancelled" if cancelled else "failed"
                self.store.update(job["id"], status=state, error=safe_error(error), finished_at=time.time())
                # A CUDA/OOM failure may poison the context. Always recreate it for the next job.
                try:
                    self.runtime.stop()
                except Exception as stop_error:
                    logging.error("Worker cleanup failed: %s", safe_error(stop_error))
                    self.shutdown.set()
            finally:
                self.current_job = None

    def close(self):
        self.shutdown.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("Scheduler did not stop; refusing concurrent worker lifecycle operations")
        self.runtime.close()
