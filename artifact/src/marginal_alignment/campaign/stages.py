"""Fail-closed, resumable stage execution for the evaluation campaign."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import canonical_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def hash_tree(root: Path, includes: Iterable[str]) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for expression in includes:
        for path in sorted(root.glob(expression)):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            records.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    unique = {record["path"]: record for record in records}
    records = [unique[key] for key in sorted(unique)]
    digest = hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()
    return digest, records


def _output_records(paths: Iterable[Path], run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            raise FileNotFoundError(f"declared output was not produced: {path}")
        try:
            label = path.relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            label = str(path)
        records.append({"path": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records


@dataclass(frozen=True)
class StageContext:
    project_root: Path
    run_dir: Path
    config: dict[str, Any]
    config_hash: str
    source_hash: str
    runtime_hash: str = "unspecified_runtime"
    external_input_hash: str = "unspecified_external_inputs"


StageFunction = Callable[[StageContext], Iterable[Path]]


class StageRunner:
    """Execute stages exactly once per input fingerprint and verify all resumes."""

    def __init__(self, context: StageContext, *, heartbeat_interval_seconds: float = 30.0):
        interval = float(heartbeat_interval_seconds)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("heartbeat_interval_seconds must be finite and positive")
        self.context = context
        self.heartbeat_interval_seconds = interval
        self.state_dir = context.run_dir / "stages"
        self.log_dir = context.run_dir / "logs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _fingerprint(self, name: str, dependencies: list[str]) -> str:
        upstream = []
        for dependency in dependencies:
            path = self.state_dir / f"{dependency}.json"
            if not path.is_file():
                raise RuntimeError(f"stage {name} requires incomplete stage {dependency}")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "PASS":
                raise RuntimeError(f"stage {name} requires passing stage {dependency}")
            upstream.append({"stage": dependency, "fingerprint": manifest["fingerprint"],
                             "outputs": manifest["outputs"]})
        payload = {
            "stage": name,
            "config_sha256": self.context.config_hash,
            "source_sha256": self.context.source_hash,
            "runtime_sha256": self.context.runtime_hash,
            "external_input_sha256": self.context.external_input_hash,
            "upstream": upstream,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def _resume_valid(self, manifest: dict[str, Any], fingerprint: str) -> bool:
        if manifest.get("status") != "PASS" or manifest.get("fingerprint") != fingerprint:
            return False
        for record in manifest.get("outputs", []):
            path = Path(record["path"])
            if not path.is_absolute():
                path = self.context.run_dir / path
            if (not path.is_file() or path.stat().st_size != int(record["bytes"])
                    or sha256_file(path) != record["sha256"]):
                raise RuntimeError(
                    f"refusing unsafe resume: output changed after PASS: {record['path']}"
                )
        return True

    def run(self, name: str, function: StageFunction, *, dependencies: list[str] | None = None,
            force: bool = False) -> dict[str, Any]:
        dependencies = dependencies or []
        fingerprint = self._fingerprint(name, dependencies)
        state_path = self.state_dir / f"{name}.json"
        if state_path.is_file() and not force:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if self._resume_valid(previous, fingerprint):
                print(f"[resume] {name}: verified PASS", flush=True)
                return previous
        started = time.time()
        running = {
            "schema_version": "marginal_alignment.campaign.stage.v2",
            "stage": name,
            "status": "RUNNING",
            "fingerprint": fingerprint,
            "started_unix": started,
            "heartbeat_unix": started,
            "heartbeat_sequence": 0,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "dependencies": dependencies,
            "outputs": [],
        }
        atomic_json(state_path, running)
        log_path = self.log_dir / f"{name}.log"
        print(f"[run] {name}", flush=True)
        heartbeat_stop = threading.Event()
        heartbeat_lock = threading.Lock()
        heartbeat_state: dict[str, float | int] = {"unix": started, "sequence": 0}
        heartbeat_errors: list[BaseException] = []

        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(self.heartbeat_interval_seconds):
                try:
                    with heartbeat_lock:
                        previous = float(heartbeat_state["unix"])
                        now = max(time.time(), math.nextafter(previous, math.inf))
                        sequence = int(heartbeat_state["sequence"]) + 1
                        heartbeat_state.update({"unix": now, "sequence": sequence})
                    # The heartbeat owns the RUNNING state file until it is
                    # stopped and joined. Terminal PASS/FAILED publication
                    # happens only after the join, so it cannot be overwritten
                    # by a late heartbeat.
                    atomic_json(state_path, {
                        **running,
                        "heartbeat_unix": now,
                        "heartbeat_sequence": sequence,
                    })
                    # Open a distinct O_APPEND handle rather than sharing the
                    # stdout redirect's TextIOWrapper across threads.
                    with log_path.open("a", encoding="utf-8") as heartbeat_log:
                        heartbeat_log.write(
                            f"[heartbeat] stage={name} sequence={sequence} unix={now:.6f}\n"
                        )
                except BaseException as error:  # surfaced by the main thread
                    heartbeat_errors.append(error)
                    heartbeat_stop.set()
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name=f"stage-heartbeat-{name}",
            daemon=True,
        )
        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                log_handle.write(f"\n=== {name} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
                heartbeat_thread.start()
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    outputs = list(function(self.context))
            # Output hashing can itself be long for feature artifacts, so the
            # RUNNING heartbeat remains live through integrity verification.
            records = _output_records(outputs, self.context.run_dir)
            heartbeat_stop.set()
            heartbeat_thread.join()
            if heartbeat_errors:
                error = heartbeat_errors[0]
                raise RuntimeError(
                    f"stage heartbeat failed: {type(error).__name__}: {error}"
                ) from error
            completed = time.time()
            with heartbeat_lock:
                latest_heartbeat = max(completed, float(heartbeat_state["unix"]))
                heartbeat_sequence = int(heartbeat_state["sequence"])
            passed = {
                **running,
                "status": "PASS",
                "heartbeat_unix": latest_heartbeat,
                "heartbeat_sequence": heartbeat_sequence,
                "completed_unix": completed,
                "elapsed_seconds": completed - started,
                "outputs": records,
                "log": log_path.relative_to(self.context.run_dir).as_posix(),
            }
            atomic_json(state_path, passed)
            print(f"[pass] {name} ({passed['elapsed_seconds']:.1f}s)", flush=True)
            return passed
        except BaseException as error:
            # Also covers failures before/while starting the thread. Joining a
            # thread that was never started raises, so inspect its identity.
            heartbeat_stop.set()
            if heartbeat_thread.ident is not None:
                heartbeat_thread.join()
            completed = time.time()
            with heartbeat_lock:
                latest_heartbeat = max(completed, float(heartbeat_state["unix"]))
                heartbeat_sequence = int(heartbeat_state["sequence"])
            failed = {
                **running,
                "status": "FAILED",
                "heartbeat_unix": latest_heartbeat,
                "heartbeat_sequence": heartbeat_sequence,
                "completed_unix": completed,
                "elapsed_seconds": completed - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "log": log_path.relative_to(self.context.run_dir).as_posix(),
            }
            atomic_json(state_path, failed)
            print(f"[failed] {name}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            raise


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "sklearn", "torch", "torchvision", "timm", "PIL"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # optional packages are reported, not hidden
            packages[name] = f"UNAVAILABLE:{type(error).__name__}"
    cuda = {"available": False}
    try:
        import torch
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception:
        pass
    # Hostname is useful operational metadata but is not part of numerical
    # reproducibility: a resumed scheduler job may legitimately land on an
    # equivalent node. Everything below runtime_identity can change numerical
    # results and is therefore separately hash-bound by the campaign.
    runtime_identity = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda": cuda,
        "thread_environment": {
            key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    return {
        "schema_version": "marginal_alignment.campaign.environment.v2",
        "host": socket.gethostname(),
        "runtime_identity": runtime_identity,
        **runtime_identity,
    }
