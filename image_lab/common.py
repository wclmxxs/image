import json
import os
import tempfile
from pathlib import Path


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path):
    return json.loads(path.read_text())


def safe_error(error):
    message = str(error)
    for key, value in os.environ.items():
        if value and len(value) >= 6 and any(word in key for word in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
            message = message.replace(value, "[redacted]")
    return f"{type(error).__name__}: {message}"[:2000]
