import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e1_progress_io as progress


class ProgressWriteTests(unittest.TestCase):
    def test_transient_denials_preserve_old_file_until_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            target.write_text("old")
            replace = os.replace
            calls = []

            def temporarily_denied(source, destination):
                calls.append(source)
                self.assertEqual(target.read_text(), "old")
                if len(calls) < 4:
                    raise PermissionError("sharing violation")
                replace(source, destination)

            with patch.object(progress.os, "replace", side_effect=temporarily_denied), patch.object(progress.time, "sleep"):
                progress.atomic_text(target, "new")
            self.assertEqual(target.read_text(), "new")
            self.assertEqual(len(calls), 4)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_persistent_denial_raises_without_corrupting_previous_state(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            target.write_text("old")
            with patch.object(progress.os, "replace", side_effect=PermissionError("denied")) as replace, \
                    patch.object(progress.time, "sleep"), self.assertRaises(PermissionError):
                progress.atomic_text(target, "new")
            self.assertEqual(replace.call_count, 20)
            self.assertEqual(target.read_text(), "old")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    @unittest.skipUnless(os.name == "nt", "Windows file-sharing behaviour")
    def test_actual_windows_reader_lock_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            target.write_text("old")
            reader = target.open()

            def release():
                time.sleep(.5)
                reader.close()

            thread = threading.Thread(target=release)
            thread.start()
            try:
                progress.atomic_text(target, "new")
            finally:
                thread.join()
                reader.close()
            self.assertEqual(target.read_text(), "new")


if __name__ == "__main__":
    unittest.main()
