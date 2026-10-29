"""Atomic progress writes with bounded retries for Windows sharing failures."""
import os
from pathlib import Path
import tempfile
import time


def atomic_text(path, text):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(.1 * 2**attempt, 1))
    finally:
        temporary.unlink(missing_ok=True)
