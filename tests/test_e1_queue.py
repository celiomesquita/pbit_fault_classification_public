from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_e1_full as queue


class QueueRecoveryTests(unittest.TestCase):
    def test_interrupted_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            interrupted = parent / "attempt_001"
            interrupted.mkdir()
            (interrupted / "trials.csv").write_text("partial result")
            completed = parent / "attempt_002"
            completed.mkdir()
            self.assertEqual(queue.next_attempt(parent), parent / "attempt_003")
            self.assertEqual((interrupted / "trials.csv").read_text(), "partial result")

    def test_completed_result_requires_untampered_outputs_and_audits(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(queue, "ROOT", Path(temporary)):
            root = Path(temporary)
            out = root / "partition"
            out.mkdir()
            (root / "scripts").mkdir()
            code = root / "scripts/verify_e1_models.py"
            code.write_text("verifier")
            data = out / "predictions.csv"
            data.write_text("preserved predictions")
            config = dict(seed=42, outer=1, scope="full_nested_training_no_outer_performance", trials=100)
            queue.write(out / "run_manifest.json", dict(config=config, outer_performance_calculated=False,
                output_sha256={data.name:queue.sha(data)}, model_fits=2096, runtime_seconds=2700))
            queue.write(out / "verification.json", dict(all_checks_passed=True, outer_performance_calculated=False,
                run_manifest_sha256=queue.sha(out / "run_manifest.json")))
            queue.write(out / "model_verification.json", dict(all_checks_passed=True, outer_performance_calculated=False,
                verification_code_sha256=queue.sha(code)))
            self.assertEqual(queue.check_verified(out, 42, 1, config)["model_fits"], 2096)
            data.write_text("altered predictions")
            with self.assertRaises(AssertionError):
                queue.check_verified(out, 42, 1, config)

    def test_progress_update_preserves_other_plan_sections(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(queue, "ROOT", Path(temporary)), \
                patch.object(queue, "OUT", Path(temporary) / "batch"):
            root = Path(temporary)
            queue.OUT.mkdir()
            (root / "docs").mkdir()
            plan = root / "docs/plano_novos_experimentos.md"
            plan.write_text("before\n" + queue.START + "\nold\n" + queue.END + "\nafter", encoding="utf-8")
            queue.publish(dict(status="running", completed=[{}], active=None), update_plan=True)
            text = plan.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("before\n"))
            self.assertTrue(text.endswith("\nafter"))
            self.assertIn("1/50", text)


if __name__ == "__main__":
    unittest.main()
