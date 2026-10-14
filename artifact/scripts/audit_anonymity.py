#!/usr/bin/env python3
"""Fail on identity, machine-path, credential, symlink, or release-clutter leaks."""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}
SKIP_NAMES = {"SHA256SUMS"}
TEXT_LIMIT = 20 * 1024 * 1024
forbidden = [
    "ath" + "arva",
    "ai" + "ren",
    "dm" + "lab",
    "/ho" + "me2/",
    "/Us" + "ers/",
    "/scr" + "atch/",
    "10." + "132.",
]
credential_patterns = [
    re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY", re.I),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
]
clutter_suffixes = (".pyc", ".pyo", ".log", ".pid", ".tar.gz")
clutter_fragments = ("failure", "handoff", "launcher")

problems: list[str] = []


def inspect_text(text: str, rel: Path, *, location: str = "") -> None:
    low = text.lower()
    where = f"{rel}{location}"
    for token in forbidden:
        if token.lower() in low:
            problems.append(f"private identity/path token in {where}: {token!r}")
    for pattern in credential_patterns:
        if pattern.search(text):
            problems.append(f"credential pattern in {where}: {pattern.pattern}")


for path in sorted(ROOT.rglob("*")):
    rel = path.relative_to(ROOT)
    if any(part in SKIP_PARTS for part in rel.parts):
        continue
    if path.is_symlink():
        problems.append(f"symlink: {rel}")
        continue
    if not path.is_file() or path.name in SKIP_NAMES:
        continue
    lowered = path.name.lower()
    if lowered.endswith(clutter_suffixes) or any(x in lowered for x in clutter_fragments):
        problems.append(f"release clutter: {rel}")
    if path.stat().st_size > 95 * 1024 * 1024:
        problems.append(f"file exceeds 95 MiB: {rel}")
    if path.suffix == ".gz":
        try:
            with gzip.open(path, mode="rt", encoding="utf-8", errors="replace") as stream:
                for line_number, line in enumerate(stream, start=1):
                    inspect_text(line, rel, location=f":{line_number}")
        except OSError as exc:
            problems.append(f"unreadable gzip member: {rel}: {exc}")
        continue
    if path.stat().st_size > TEXT_LIMIT:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    inspect_text(text, rel)

if problems:
    print("ANONYMITY AUDIT: FAIL", file=sys.stderr)
    for problem in problems:
        print(f"- {problem}", file=sys.stderr)
    raise SystemExit(1)
print("ANONYMITY AUDIT: PASS")
