#!/usr/bin/env python3
"""Build a deterministic SHA-256 inventory for the public artifact."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS"
SKIP = {".git", "__pycache__", ".pytest_cache"}
rows: list[str] = []
for path in sorted(ROOT.rglob("*")):
    rel = path.relative_to(ROOT)
    if not path.is_file() or path == OUTPUT or any(part in SKIP for part in rel.parts):
        continue
    rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel.as_posix()}")
OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"Wrote {len(rows)} entries to {OUTPUT.name}")

