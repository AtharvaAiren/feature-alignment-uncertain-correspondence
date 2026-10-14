from __future__ import annotations

import ast
from pathlib import Path


def test_namespace_has_no_duplicate_literal_dictionary_keys() -> None:
    """Prevent Python's silent last-value-wins behavior in result schemas."""

    source_root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "marginal_alignment"
        / "paired_recovery_bounds"
    )
    duplicates: list[str] = []
    for path in sorted(source_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            literal_keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            repeated = sorted({key for key in literal_keys if literal_keys.count(key) > 1})
            duplicates.extend(
                f"{path.name}:{node.lineno}:{key}" for key in repeated
            )
    assert duplicates == []
