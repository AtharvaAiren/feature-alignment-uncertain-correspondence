import json
import time

import pytest

from marginal_alignment.campaign.stages import StageContext, StageRunner


def _context(tmp_path):
    return StageContext(tmp_path, tmp_path / "run", {}, "c" * 64, "s" * 64)


def test_stage_resumes_only_verified_output(tmp_path):
    context = _context(tmp_path)
    runner = StageRunner(context)
    calls = []

    def stage(ctx):
        calls.append(1)
        output = ctx.run_dir / "value.txt"
        output.write_text("sealed", encoding="utf-8")
        return [output]

    runner.run("one", stage)
    runner.run("one", stage)
    assert len(calls) == 1
    (context.run_dir / "value.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe resume"):
        runner.run("one", stage)


def test_stage_failure_is_visible_and_dependency_fails_closed(tmp_path):
    context = _context(tmp_path)
    runner = StageRunner(context)

    def broken(_):
        raise ValueError("intentional")

    with pytest.raises(ValueError, match="intentional"):
        runner.run("broken", broken)
    state = json.loads((context.run_dir / "stages/broken.json").read_text())
    assert state["status"] == "FAILED"
    with pytest.raises(RuntimeError, match="requires passing"):
        runner.run("next", lambda _: [], dependencies=["broken"])


def test_stage_heartbeat_advances_and_stops_before_terminal_manifest(tmp_path):
    context = _context(tmp_path)
    runner = StageRunner(context, heartbeat_interval_seconds=0.02)
    observed: list[tuple[int, float]] = []

    def long_stage(ctx):
        state_path = ctx.run_dir / "stages/long.json"
        initial = json.loads(state_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if int(state["heartbeat_sequence"]) > int(initial["heartbeat_sequence"]):
                observed.append((int(state["heartbeat_sequence"]), float(state["heartbeat_unix"])))
                break
            time.sleep(0.005)
        assert observed, "heartbeat did not become externally visible while stage was running"
        output = ctx.run_dir / "long.txt"
        output.write_text("complete", encoding="utf-8")
        return [output]

    manifest = runner.run("long", long_stage)
    assert manifest["status"] == "PASS"
    assert manifest["heartbeat_sequence"] >= 1
    assert manifest["heartbeat_unix"] >= observed[-1][1]
    state_path = context.run_dir / "stages/long.json"
    terminal_contents = state_path.read_bytes()
    log_path = context.run_dir / "logs/long.log"
    terminal_log = log_path.read_bytes()
    time.sleep(0.08)
    assert state_path.read_bytes() == terminal_contents
    assert log_path.read_bytes() == terminal_log
    assert b"[heartbeat] stage=long" in terminal_log


def test_stage_heartbeat_stops_cleanly_on_failure(tmp_path):
    context = _context(tmp_path)
    runner = StageRunner(context, heartbeat_interval_seconds=0.01)

    def delayed_failure(_):
        time.sleep(0.04)
        raise ValueError("after heartbeat")

    with pytest.raises(ValueError, match="after heartbeat"):
        runner.run("delayed_failure", delayed_failure)
    state_path = context.run_dir / "stages/delayed_failure.json"
    failed = json.loads(state_path.read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED"
    assert failed["heartbeat_sequence"] >= 1
    terminal_contents = state_path.read_bytes()
    time.sleep(0.05)
    assert state_path.read_bytes() == terminal_contents


def test_stage_rejects_invalid_heartbeat_interval(tmp_path):
    with pytest.raises(ValueError, match="finite and positive"):
        StageRunner(_context(tmp_path), heartbeat_interval_seconds=0.0)


def test_external_input_identity_invalidates_stage_resume(tmp_path):
    base = _context(tmp_path)
    first = StageContext(
        base.project_root, base.run_dir, base.config, base.config_hash,
        base.source_hash, base.runtime_hash, "a" * 64,
    )
    calls = []

    def stage(context):
        calls.append(context.external_input_hash)
        output = context.run_dir / "external-bound.txt"
        output.write_text(context.external_input_hash, encoding="utf-8")
        return [output]

    StageRunner(first, heartbeat_interval_seconds=0.01).run("cache_bound", stage)
    StageRunner(first, heartbeat_interval_seconds=0.01).run("cache_bound", stage)
    second = StageContext(
        first.project_root, first.run_dir, first.config, first.config_hash,
        first.source_hash, first.runtime_hash, "b" * 64,
    )
    StageRunner(second, heartbeat_interval_seconds=0.01).run("cache_bound", stage)
    assert calls == ["a" * 64, "b" * 64]
