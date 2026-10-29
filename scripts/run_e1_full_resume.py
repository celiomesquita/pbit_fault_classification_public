"""Resume the original E1 queue with a separately frozen progress-I/O repair."""
import argparse
import json

import run_e1_full as queue
from e1_progress_io import atomic_text

AMENDMENT = queue.OUT / "resume_001_freeze.json"
original_check_sources = queue.check_sources


def check_resume_sources():
    frozen = original_check_sources()
    amended = queue.read(AMENDMENT)
    assert amended["batch_freeze_sha256"] == queue.sha(queue.OUT / "batch_freeze.json")
    for name, expected in amended["source_sha256"].items():
        assert queue.sha(queue.ROOT / name) == expected, name
    return frozen


def prepare():
    assert not AMENDMENT.exists(), "Resume amendment already frozen."
    original = original_check_sources()
    state = queue.read(queue.OUT / "progress.json")
    assert state["status"] == "interrompida por falha"
    for result in state["completed"]:
        queue.check_verified(queue.ROOT / result["output"], result["seed"], result["outer"], original["config"])
    active = state["active"]
    recovered = queue.check_verified(queue.ROOT / active["output"], active["seed"], active["outer"], original["config"])
    paths = ["scripts/e1_progress_io.py", "scripts/run_e1_full_resume.py",
             "tests/test_e1_progress_io.py", "docs/protocolos/E1_retomada_gravacao.md"]
    snapshot = queue.OUT / "resume_001_snapshots"
    snapshot.mkdir()
    for name in ["progress.json", "progress.md", "queue.stdout.log", "queue.stderr.log"]:
        (snapshot / name).write_bytes((queue.OUT / name).read_bytes())
    (snapshot / "plan_before_resume.md").write_bytes((queue.ROOT / "docs/plano_novos_experimentos.md").read_bytes())
    for name in paths:
        target = snapshot / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((queue.ROOT / name).read_bytes())
    atomic_text(AMENDMENT, json.dumps(dict(
        frozen_at_utc=queue.now(), batch_freeze_sha256=queue.sha(queue.OUT / "batch_freeze.json"),
        source_sha256={name:queue.sha(queue.ROOT / name) for name in paths},
        recovered_partition=recovered,
        snapshot_sha256={p.relative_to(queue.OUT).as_posix():queue.sha(p) for p in snapshot.rglob("*") if p.is_file()},
        scientific_configuration_changed=False, outer_performance_calculated=False), indent=2) + "\n")
    print("Progress-I/O amendment frozen; original code and failure history preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "run"], required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        queue.atomic_text = atomic_text
        queue.check_sources = check_resume_sources
        queue.run_queue()
        queue.write(queue.OUT / "resume_completion_manifest.json", dict(
            completed_at_utc=queue.now(), resume_freeze_sha256=queue.sha(AMENDMENT),
            original_completion_sha256=queue.sha(queue.OUT / "completion_manifest.json"),
            outer_performance_calculated=False, inference_pending=True))
