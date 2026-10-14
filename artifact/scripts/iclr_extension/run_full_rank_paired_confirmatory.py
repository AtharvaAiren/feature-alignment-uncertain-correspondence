#!/usr/bin/env python3
"""Run the additive untruncated paired-ridge confirmatory follow-up."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


_TRUST_DIRECTORY = Path("provenance/full_rank")
_BASE_LOCK = _TRUST_DIRECTORY / "BASE_DEPENDENCY_SHA256SUMS"
_PAYLOAD_LOCK = _TRUST_DIRECTORY / "PAYLOAD_SHA256SUMS"
_BASE_PATHS = {
    "src/marginal_alignment/__init__.py",
    "src/marginal_alignment/ceiling.py",
    "src/marginal_alignment/linalg.py",
    "src/marginal_alignment/metrics.py",
    "src/marginal_alignment/transforms.py",
    "src/marginal_alignment/campaign/__init__.py",
    "src/marginal_alignment/campaign/artifacts.py",
    "src/marginal_alignment/campaign/cache_import.py",
    "src/marginal_alignment/campaign/feature_protocol.py",
    "src/marginal_alignment/campaign/feature_store.py",
    "src/marginal_alignment/campaign/inference.py",
    "src/marginal_alignment/campaign/maps.py",
    "src/marginal_alignment/campaign/method_registry.py",
    "src/marginal_alignment/campaign/metrics.py",
    "src/marginal_alignment/campaign/real_runner.py",
    "src/marginal_alignment/campaign/raw_extract.py",
    "src/marginal_alignment/campaign/split_plan.py",
    "src/marginal_alignment/iclr_extension/__init__.py",
    "src/marginal_alignment/iclr_extension/ceiling.py",
    "src/marginal_alignment/iclr_extension/frontier.py",
    "src/marginal_alignment/iclr_extension/natural_shift.py",
}
_FULL_RANK_PAYLOAD_PATHS = {
    "configs/iclr_extension/full_rank_paired_confirmatory_lock.json",
    "scripts/iclr_extension/run_full_rank_paired_confirmatory.py",
    "src/marginal_alignment/iclr_extension/full_rank_confirmatory.py",
    _BASE_LOCK.as_posix(),
}


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_lock(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"source checksum lock is absent or symbolic: {path}")
    result: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line or len(line) < 67 or line[64:66] != "  ":
            raise RuntimeError(f"malformed source checksum line {path}:{line_number}")
        digest, relative = line[:64], line[66:]
        relative_path = Path(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative_path.as_posix() != relative
            or relative in result
        ):
            raise RuntimeError(f"unsafe source checksum line {path}:{line_number}")
        result[relative] = digest
    if not result:
        raise RuntimeError(f"source checksum lock is empty: {path}")
    return result


def _preimport_source_trust(project_root: Path) -> None:
    """Verify the deployed source closure before importing any project code."""

    for relative in (_BASE_LOCK, _PAYLOAD_LOCK):
        raw = project_root / relative
        if raw.is_symlink() or not raw.is_file():
            raise RuntimeError(f"source checksum lock is absent or symbolic: {raw}")
        try:
            raw.resolve(strict=True).relative_to(project_root)
        except ValueError as error:
            raise RuntimeError(
                f"source checksum lock escapes project root: {raw}"
            ) from error
    locks = {
        "base": _read_lock(project_root / _BASE_LOCK),
        "payload": _read_lock(project_root / _PAYLOAD_LOCK),
    }
    if set(locks["base"]) != _BASE_PATHS:
        raise RuntimeError("base dependency checksum inventory drifted")
    missing = sorted(_FULL_RANK_PAYLOAD_PATHS - set(locks["payload"]))
    if missing:
        raise RuntimeError(f"payload checksum lock lacks paired-ridge files: {missing}")
    for label, records in locks.items():
        for relative, expected in records.items():
            candidate = project_root / relative
            if candidate.is_symlink() or not candidate.is_file():
                raise RuntimeError(f"{label} source is absent or symbolic: {relative}")
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(project_root)
            except ValueError as error:
                raise RuntimeError(
                    f"{label} source escapes project root: {relative}"
                ) from error
            observed = _checksum(resolved)
            if observed != expected:
                raise RuntimeError(
                    f"{label} source SHA-256 mismatch: {relative}: {observed} != {expected}"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--original-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="authenticate frozen inputs and feature stores without creating output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    project_root = arguments.project_root.expanduser().resolve(strict=True)
    _preimport_source_trust(project_root)
    sys.path.insert(0, str(project_root / "src"))
    from marginal_alignment.iclr_extension.full_rank_confirmatory import (
        load_and_validate_inputs,
        record_failure,
        require_single_thread_environment,
        run_extension,
    )

    require_single_thread_environment()
    inputs = load_and_validate_inputs(
        project_root=project_root,
        original_root=arguments.original_run_root,
        verify_store_files=True,
    )
    print(
        "PRE-OUTCOME IDENTITY PASS: "
        f"{len(inputs.records)} feature stores, {len(inputs.anchors)} Bures anchors, "
        f"{len(inputs.descriptive_paired_reduced_rank)} paired-reduced-rank descriptive rows, "
        "split/evaluation/store hashes matched across all original confirmatory methods",
        flush=True,
    )
    if arguments.preflight_only:
        print("Preflight-only mode: no follow-up output was created", flush=True)
        return 0
    try:
        done = run_extension(
            inputs=inputs,
            output_root=arguments.output_root,
            workers=arguments.workers,
        )
    except Exception as error:
        try:
            failure = record_failure(
                arguments.output_root,
                error,
                original_root=inputs.original_root,
            )
        except Exception as record_error:
            print(
                f"Failure record rejected: {type(record_error).__name__}: {record_error}",
                file=sys.stderr,
            )
        else:
            if failure is not None:
                print(f"Failure record: {failure}", file=sys.stderr)
        raise
    print(f"UNTRUNCATED PAIRED-RIDGE CONFIRMATORY FOLLOW-UP PASS: {done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
