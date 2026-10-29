"""Durable serial queue for E1; audit each partition without outer scoring."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/e1_full_v1"
SEEDS = [42, 0, 7, 13, 21, 1, 2, 3, 4, 5]
START = "<!-- E1_FULL_PROGRESS_START -->"
END = "<!-- E1_FULL_PROGRESS_END -->"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def check_sources():
    frozen = read(OUT / "batch_freeze.json")
    for name, expected in frozen["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert sys.version == frozen["python"]
    for package, expected in frozen["environment"].items():
        assert importlib.metadata.version(package) == expected, package
    return frozen


def check_verified(directory, seed, outer, config):
    manifest = read(directory / "run_manifest.json")
    expected = dict(config, seed=seed, outer=outer)
    actual = dict(manifest["config"])
    assert actual["scope"] in [config["scope"], "full_budget_timing_pilot_no_outer_performance"]
    actual["scope"] = expected["scope"]
    assert actual == expected and not manifest["outer_performance_calculated"]
    for name, expected_hash in manifest["output_sha256"].items():
        assert sha(directory / name) == expected_hash, name
    verified = read(directory / "verification.json")
    assert verified["all_checks_passed"] and not verified["outer_performance_calculated"]
    assert verified["run_manifest_sha256"] == sha(directory / "run_manifest.json")
    model_check = read(directory / "model_verification.json")
    assert model_check["all_checks_passed"] and not model_check["outer_performance_calculated"]
    assert model_check["verification_code_sha256"] == sha(ROOT / "scripts/verify_e1_models.py")
    return dict(seed=seed, outer=outer, output=directory.relative_to(ROOT).as_posix(),
                run_manifest_sha256=sha(directory / "run_manifest.json"),
                verification_sha256=sha(directory / "verification.json"),
                model_verification_sha256=sha(directory / "model_verification.json"),
                model_fits=manifest["model_fits"], runtime_seconds=manifest["runtime_seconds"])


def next_attempt(parent):
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while (parent / f"attempt_{number:03d}").exists():
        number += 1
    return parent / f"attempt_{number:03d}"


def publish(state, update_plan=False):
    state["updated_at_utc"] = now()
    write(OUT / "progress.json", state)
    active = state.get("active")
    stage = state["status"]
    detail = ""
    if active:
        detail = f"; semente {active['seed']}, divisão {active['outer']}, etapa {active['stage']}"
    body = (f"E1: {len(state['completed'])}/50 divisões verificadas; estado {stage}{detail}.\n"
            f"Actualização UTC: {state['updated_at_utc']}. Inferência após E3 ainda pendente.\n")
    counts = active.get("completed_trials", {}) if active else {}
    trials = "\nTentativas por busca: " + "; ".join(f"{arm}: {count}/100" for arm, count in sorted(counts.items())) + ".\n" if counts else ""
    atomic_text(OUT / "progress.md", "# Acompanhamento da rodada E1\n\n" + body + trials)
    if update_plan:
        plan = ROOT / "docs/plano_novos_experimentos.md"
        text = plan.read_text(encoding="utf-8")
        assert text.count(START) == text.count(END) == 1
        before, rest = text.split(START)
        _, after = rest.split(END)
        atomic_text(plan, before + START + "\n" + body + END + after)


def prepare():
    assert not OUT.exists(), "Existing queue: use --mode run or status."
    pilot = ROOT / "data/e1_pilot"
    original = read(pilot / "protocol_freeze.json")
    for name, digest in original["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    config = dict(original["config"], scope="full_nested_training_no_outer_performance")
    adopted = check_verified(pilot, 42, 0, config)
    OUT.mkdir()
    paths = list(original["source_sha256"]) + [
        "docs/protocolos/E1_rodada_completa.md", "scripts/run_e1_partition.py",
        "scripts/run_e1_full.py", "scripts/verify_e1_models.py", "tests/test_e1_queue.py"]
    for name in paths:
        source = ROOT / name
        if source.suffix in [".md", ".py"]:
            snapshot = OUT / "frozen_sources" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source.read_bytes())
    write(OUT / "batch_freeze.json", dict(frozen_at_utc=now(), config=config,
          seeds=SEEDS, outer_folds=5, workers=4, environment=original["environment"],
          python=sys.version, source_sha256={name:sha(ROOT / name) for name in paths},
          adopted_pilot=adopted, pilot_protocol_freeze_sha256=sha(pilot / "protocol_freeze.json"),
          plan_at_start_sha256=sha(ROOT / "docs/plano_novos_experimentos.md")))
    (OUT / "plan_at_start.md").write_bytes((ROOT / "docs/plano_novos_experimentos.md").read_bytes())
    publish(dict(status="preparada", completed=[adopted], active=None,
                 outer_performance_calculated=False), update_plan=True)
    print("Queue frozen; verified pilot adopted; 49 partitions pending.", flush=True)


def execute(command, log, state, stage):
    check_sources()
    state["active"]["stage"] = stage
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log.open("ab") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=flags)
        state["active"]["child_pid"] = child.pid
        publish(state, update_plan=True)
        while child.poll() is None:
            time.sleep(10)
            counts = {}
            directory = ROOT / state["active"]["output"]
            for path in directory.glob("search/*/trials.csv"):
                with path.open(newline="", encoding="utf-8") as source:
                    counts[path.parent.name] = max(0, sum(1 for _ in csv.reader(source))-1)
            state["active"]["completed_trials"] = counts
            publish(state)
        assert child.returncode == 0, f"{stage} exited {child.returncode}; see {log}"


def run_queue():
    frozen = check_sources()
    # A crashed Python process releases the Windows byte-range lock automatically.
    import msvcrt
    with (OUT / "queue.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = read(OUT / "progress.json")
        old_active = state.get("active")
        if old_active and old_active.get("child_pid"):
            # An orphan child must end before another queue can start the same fold.
            probe = subprocess.run(["powershell", "-NoProfile", "-Command",
                f"if (Get-Process -Id {int(old_active['child_pid'])} -ErrorAction SilentlyContinue) {{ exit 1 }}"],
                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            assert probe.returncode == 0, "Previous child still running; do not duplicate it."
        state["completed"] = [check_verified(ROOT / c["output"], c["seed"], c["outer"], frozen["config"])
                              for c in state["completed"]]
        state.update(status="em execução", runner_pid=os.getpid(), active=None)
        publish(state, update_plan=True)
        try:
            for seed in frozen["seeds"]:
                for outer in range(5):
                    if any(c["seed"] == seed and c["outer"] == outer for c in state["completed"]):
                        continue
                    parent = OUT / "partitions" / f"seed_{seed}_outer_{outer}"
                    # Adopt a fully audited result left by a crash before progress was written.
                    reusable = None
                    for previous in sorted(parent.glob("attempt_*")):
                        if (previous / "verification.json").exists() and (previous / "model_verification.json").exists():
                            reusable = check_verified(previous, seed, outer, frozen["config"])
                            break
                    if reusable:
                        state["completed"].append(reusable)
                        publish(state, update_plan=True)
                        continue
                    output = next_attempt(parent)
                    relative = output.relative_to(ROOT).as_posix()
                    state["active"] = dict(seed=seed, outer=outer, output=relative, stage="treino")
                    log = parent / f"{output.name}.log"
                    print(f"Starting seed={seed} outer={outer}; {len(state['completed'])}/50 verified.", flush=True)
                    execute([sys.executable, "-u", "scripts/run_e1_partition.py", "--seed", str(seed),
                             "--outer", str(outer), "--output", relative], log, state, "treino")
                    execute([sys.executable, "scripts/verify_e1_pilot.py", "--output", relative], log, state, "conferência")
                    execute([sys.executable, "scripts/verify_e1_models.py", "--output", relative], log, state, "reprodução")
                    state["completed"].append(check_verified(output, seed, outer, frozen["config"]))
                    state["active"] = None
                    publish(state, update_plan=True)
                    print(f"Verified seed={seed} outer={outer}; {len(state['completed'])}/50 complete.", flush=True)
            check_sources()
            state.update(status="treinos e conferências concluídos", active=None)
            publish(state, update_plan=True)
            write(OUT / "completion_manifest.json", dict(completed_at_utc=now(),
                  batch_freeze_sha256=sha(OUT / "batch_freeze.json"), completed=state["completed"],
                  outer_performance_calculated=False, inference_pending=True))
        except BaseException as error:
            state.update(status="interrompida por falha", error=str(error), traceback=traceback.format_exc())
            publish(state, update_plan=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "run", "status"], default="status")
    mode = parser.parse_args().mode
    if mode == "prepare":
        prepare()
    elif mode == "run":
        run_queue()
    else:
        print((OUT / "progress.md").read_text(encoding="utf-8"))
