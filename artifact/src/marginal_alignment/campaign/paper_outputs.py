"""Evidence-only paper tables and figures for the evaluation campaign.

Every panel is derived from persisted campaign artifacts.  Missing secondary
suites produce explicit ``UNAVAILABLE`` JSON/Markdown panels; they never produce
placeholder measurements.  The primary marginal endpoint is fixed at n=4096,
and method rows are discovered from evidence rather than a wish-list.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import IntegrityError, read_json, sha256_file, write_json_atomic


PRIMARY_CALIBRATION_SIZE = 4096
PANEL_SCHEMA = "marginal_alignment.campaign.paper_panel.v2"

DISPLAY = {
    "none": "No alignment",
    "mean": "Mean",
    "diagonal": "Diagonal / BN-style",
    "quantile": "Coordinate quantile",
    "coral_oas": "CORAL (OAS)",
    "bures_oas": "Bures (OAS)",
    "bures_pair_endpoints": "Bures (pair endpoints; assignment erased)",
    "bures_lw": "Bures (Ledoit--Wolf)",
    "bures_empirical": "Bures (empirical)",
    "bures_rank4_oas": "Bures rank 4 (OAS)",
    "bures_rank16_oas": "Bures rank 16 (OAS)",
    "bures_rank64_oas": "Bures rank 64 (OAS)",
    "bures_rank256_oas": "Bures rank 256 (OAS)",
    "sliced_quantile_r64": "Sliced quantile",
    "third_moment_ica_r64": "Third-moment ICA",
    "paired_procrustes": "Paired Procrustes",
    "paired_procrustes_shuffled": "Paired Procrustes (shuffled)",
    "paired_ridge": "Paired ridge",
    "paired_ridge_shuffled": "Paired ridge (shuffled)",
    "paired_reduced_rank": "Paired reduced-rank ridge",
    "paired_reduced_rank_shuffled": "Paired reduced-rank ridge (shuffled)",
    "paired_readout_gauge": "Paired readout gauge",
    "paired_readout_gauge_shuffled": "Paired readout gauge (shuffled)",
    "labeled_er": "Labeled ER",
    "identity_canonical": "Fixed identity",
    "second_moment_permutation": "First/second moments",
    "third_moment_permutation": "Third moments",
    "likelihood_marginal_oracle": "Marginal likelihood oracle",
    "paired_permutation": "Paired permutation diagnostic",
}

METHOD_ORDER = (
    "none", "mean", "diagonal", "quantile", "coral_oas", "bures_empirical",
    "bures_lw", "bures_rank4_oas", "bures_rank16_oas", "bures_rank64_oas", "bures_rank256_oas",
    "bures_oas", "bures_pair_endpoints", "third_moment_ica_r64", "sliced_quantile_r64",
    "paired_procrustes", "paired_procrustes_shuffled", "paired_ridge",
    "paired_ridge_shuffled", "paired_reduced_rank", "paired_reduced_rank_shuffled",
    "paired_readout_gauge", "paired_readout_gauge_shuffled", "labeled_er",
)
METHOD_RANK = {name: index for index, name in enumerate(METHOD_ORDER)}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, contents: str) -> Path:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise IntegrityError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _source_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_real_inputs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    rows_path = root / "real/rows.jsonl"
    summary_path = root / "real/summary.json"
    manifest_path = root / "real/manifest.json"
    if not all(path.is_file() for path in (rows_path, summary_path, manifest_path)):
        raise FileNotFoundError("real rows, summary, and manifest are required for paper outputs")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise IntegrityError("real manifest is not PASS")
    if manifest.get("rows_sha256") != sha256_file(rows_path):
        raise IntegrityError("real rows do not match their persisted manifest")
    if manifest.get("summary_sha256") != sha256_file(summary_path):
        raise IntegrityError("real summary does not match its persisted manifest")
    rows = _read_jsonl(rows_path)
    summary = read_json(summary_path)
    groups = summary.get("groups")
    if not isinstance(groups, list):
        raise IntegrityError("real summary has no groups")
    if int(manifest.get("record_count", -1)) != len(rows):
        raise IntegrityError("real record count does not match rows.jsonl")
    return rows, groups, [rows_path, summary_path, manifest_path]


def _method_key(name: Any) -> tuple[int, str]:
    label = str(name)
    base = re.sub(r"_noise[-+0-9.eE]+$", "", label)
    return METHOD_RANK.get(base, len(METHOD_RANK)), label


def _sorted_rows(rows: Iterable[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        result = []
        for name in keys:
            value = row.get(name)
            result.append(_method_key(value) if name == "method" else (value is None, str(value)))
        return tuple(result)
    return sorted(rows, key=key)


def _macro_aggregate(
    rows: Iterable[dict[str, Any]], *, keys: Sequence[str], metrics: Sequence[str],
    repeat_key: str = "split_seed",
) -> list[dict[str, Any]]:
    """Average conditions within repeat, then report mean/SD across repeats."""

    first: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        first[tuple(row.get(key) for key in (*keys, repeat_key))].append(row)
    split_rows = []
    for key, values in first.items():
        record = {name: value for name, value in zip((*keys, repeat_key), key)}
        record["condition_cells"] = len(values)
        for metric in metrics:
            vector = [float(row[metric]) for row in values if _finite(row.get(metric))]
            record[metric] = float(np.mean(vector)) if vector else None
        split_rows.append(record)
    second: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows:
        second[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for key, values in second.items():
        record = {name: value for name, value in zip(keys, key)}
        record["repeats"] = len(values)
        record["repeat_key"] = repeat_key
        record["condition_cells"] = int(sum(row["condition_cells"] for row in values))
        record["variation_unit"] = f"SD across {repeat_key}-level condition-macro means"
        for metric in metrics:
            vector = np.asarray([row[metric] for row in values if _finite(row.get(metric))])
            record[f"{metric}_mean"] = float(vector.mean()) if len(vector) else None
            record[f"{metric}_std"] = (
                float(vector.std(ddof=1)) if len(vector) > 1 else 0.0 if len(vector) else None
            )
        output.append(record)
    return _sorted_rows(output, [*keys])


def _format_value(value: Any, *, percent: bool = False, digits: int = 3) -> str:
    if not _finite(value):
        return "--"
    number = float(value) * (100.0 if percent else 1.0)
    return f"{number:.{digits}f}"


def _format_pm(row: Mapping[str, Any], metric: str, *, percent: bool = False) -> str:
    mean, std = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
    if not _finite(mean):
        return "--"
    return f"{_format_value(mean, percent=percent)} ± {_format_value(std, percent=percent)}"


Column = tuple[str, str, Callable[[Mapping[str, Any]], str]]


def _write_panel(
    output: Path, *, name: str, title: str, rows: list[dict[str, Any]],
    columns: Sequence[Column], sources: Sequence[Path], root: Path,
    notes: Sequence[str] = (), status: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    panel_status = status or ("PASS" if rows else "UNAVAILABLE")
    reason = None if rows else "No persisted rows matched this prespecified panel."
    payload = {
        "schema_version": PANEL_SCHEMA,
        "name": name,
        "title": title,
        "status": panel_status,
        "reason": reason,
        "row_count": len(rows),
        "methods_present": sorted(
            {str(row["method"]) for row in rows if row.get("method") is not None},
            key=_method_key,
        ),
        "notes": list(notes),
        "sources": [_source_record(root, path) for path in sources if path.is_file()],
        "rows": _json_safe(rows),
    }
    json_path = output / f"{name}.json"
    markdown_path = output / f"{name}.md"
    write_json_atomic(json_path, payload)
    lines = [f"# {title}", "", f"Status: `{panel_status}`. Evidence rows: {len(rows)}.", ""]
    if reason:
        lines.extend([reason, ""])
    for note in notes:
        lines.extend([f"- {note}", ""])
    if rows:
        lines.extend([
            "| " + " | ".join(label for _, label, _ in columns) + " |",
            "|" + "|".join("---" if index < len(columns) - 1 else "---"
                             for index in range(len(columns))) + "|",
        ])
        for row in rows:
            lines.append("| " + " | ".join(formatter(row) for _, _, formatter in columns) + " |")
        lines.append("")
    _atomic_text(markdown_path, "\n".join(lines))
    return json_path, markdown_path, payload


def _label(name: Any) -> str:
    return DISPLAY.get(str(name), str(name).replace("_", " "))


def _save_figure(figure: Any, output: Path, name: str) -> list[Path]:
    generated = []
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        path = output / f"{name}.{suffix}"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        generated.append(path)
    return generated


def _plot_method_lines(
    rows: list[dict[str, Any]], output: Path, name: str, *, x_key: str,
    y_metric: str, xlabel: str, ylabel: str, percent: bool = True,
    group_keys: Sequence[str] = ("dataset", "encoder", "method"), log_x: bool = False,
) -> list[Path]:
    rows = [row for row in rows if _finite(row.get(x_key))
            and _finite(row.get(f"{y_metric}_mean"))]
    if not rows:
        return []
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        values = sorted(values, key=lambda row: float(row[x_key]))
        x = [float(row[x_key]) for row in values]
        scale = 100.0 if percent else 1.0
        y = [scale * float(row[f"{y_metric}_mean"]) for row in values]
        label_parts = [_label(value) if group_keys[index] == "method" else str(value)
                       for index, value in enumerate(key)]
        axis.plot(x, y, marker="o", linewidth=1.4, label=" · ".join(label_parts))
    if log_x:
        axis.set_xscale("log", base=2)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=6, ncol=2)
    paths = _save_figure(figure, output, name)
    plt.close(figure)
    return paths


def _plot_main(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    usable = [row for row in rows if _finite(row.get("aligned_covariance_relative_error_mean"))
              and _finite(row.get("reference_prediction_agreement_mean"))]
    if not usable:
        return []
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for row in usable:
        axis.scatter(float(row["aligned_covariance_relative_error_mean"]),
                     100 * float(row["reference_prediction_agreement_mean"]), s=28)
        axis.annotate(
            f"{row['dataset']}·{row['encoder']}·{_label(row['method'])}",
            (float(row["aligned_covariance_relative_error_mean"]),
             100 * float(row["reference_prediction_agreement_mean"])),
            fontsize=5, xytext=(2, 2), textcoords="offset points",
        )
    axis.set_xlabel("Held-out covariance residual (lower is better)")
    axis.set_ylabel("Frozen-reference prediction agreement (%)")
    axis.set_title(f"Marginal information ladder at n={PRIMARY_CALIBRATION_SIZE}")
    axis.grid(alpha=0.2)
    paths = _save_figure(figure, output, "main_marginal_ladder")
    plt.close(figure)
    return paths


def _plot_inference(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    usable = [row for row in rows if all(_finite(row.get(key))
              for key in ("effect", "ci_lower", "ci_upper"))]
    if not usable:
        return []
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.4, max(3.2, 0.34 * len(usable))))
    y = np.arange(len(usable))
    effect = np.asarray([row["effect"] for row in usable], dtype=float)
    lower = np.asarray([row["ci_lower"] for row in usable], dtype=float)
    upper = np.asarray([row["ci_upper"] for row in usable], dtype=float)
    axis.errorbar(effect, y, xerr=np.vstack([effect - lower, upper - effect]), fmt="o")
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(y, [f"{row.get('dataset', 'all')}: {_label(row['left'])} − {_label(row['right'])}"
                        for row in usable], fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("Paired effect (left − right)")
    axis.grid(axis="x", alpha=0.2)
    paths = _save_figure(figure, output, "primary_inference")
    plt.close(figure)
    return paths


def _plot_synthetic_exact(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    usable = [row for row in rows if _finite(row.get("theoretical_equal_prior_bayes_accuracy"))
              and _finite(row.get("empirical_equal_prior_bayes_accuracy"))]
    if not usable:
        return []
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.8, 4.5))
    for family in sorted({row["family"] for row in usable}):
        values = [row for row in usable if row["family"] == family]
        axis.scatter([row["theoretical_equal_prior_bayes_accuracy"] for row in values],
                     [row["empirical_equal_prior_bayes_accuracy"] for row in values],
                     s=18, alpha=0.7, label=str(family))
    axis.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Exact ceiling")
    axis.set_ylabel("Monte Carlo value")
    axis.legend(frameon=False, fontsize=7)
    axis.grid(alpha=0.2)
    paths = _save_figure(figure, output, "synthetic_exact")
    plt.close(figure)
    return paths


def _plot_synthetic_approximate(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    normalized = [{**row, "calibration_samples_plot": max(0.5, float(row["calibration_samples"]))}
                  for row in rows if _finite(row.get("exact_equal_prior_bayes_accuracy_mean"))]
    return _plot_method_lines(
        normalized, output, "synthetic_approximate", x_key="calibration_samples_plot",
        y_metric="exact_equal_prior_bayes_accuracy", xlabel="Calibration observations (0 shown at 0.5)",
        ylabel="Exact task ceiling (%)", group_keys=("epsilon", "task_disagreement"), log_x=True,
    )


def _plot_orbits(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    if not rows:
        return []
    import matplotlib.pyplot as plt

    usable = [row for row in rows if _finite(row.get("unrestricted_equal_prior_bayes_prediction_upper"))
              and _finite(row.get("achievable_group_map_randomized_minimax"))]
    if not usable:
        return []
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(usable))
    width = 0.38
    axis.bar(x - width / 2, [100 * row["unrestricted_equal_prior_bayes_prediction_upper"]
                            for row in usable], width, label="Unrestricted Bayes upper")
    axis.bar(x + width / 2, [100 * row["achievable_group_map_randomized_minimax"]
                            for row in usable], width, label="Achievable group-map minimax")
    axis.set_xticks(x, [f"{row.get('dataset')}·{row.get('encoder')}" for row in usable],
                    rotation=30, ha="right", fontsize=7)
    axis.set_ylabel("Agreement (%)")
    axis.legend(frameon=False, fontsize=7)
    axis.grid(axis="y", alpha=0.2)
    paths = _save_figure(figure, output, "controlled_real_orbit")
    plt.close(figure)
    return paths


def _load_optional_json(path: Path, *, require_pass: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, dict):
        raise IntegrityError(f"optional artifact is not a JSON object: {path}")
    if require_pass and value.get("status") != "PASS":
        raise IntegrityError(f"optional persisted artifact is not PASS: {path}")
    return value


def _load_synthetic_sources(root: Path) -> tuple[dict[str, Any] | None, list[Path]]:
    """Load a complete synthetic suite or report it absent; partial evidence fails."""

    directory = root / "synthetic"
    expected = {
        "summary.json", "manifest.json", "exact_indistinguishable_rows.jsonl",
        "approximate_two_world_rows.jsonl", "identifiable_nongaussian_rows.jsonl",
    }
    present = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    if not (present & expected):
        return None, []
    missing = expected - present
    if missing:
        raise IntegrityError(f"partial synthetic suite; missing {sorted(missing)}")
    summary_path = directory / "summary.json"
    manifest_path = directory / "manifest.json"
    summary = _load_optional_json(summary_path)
    manifest = _load_optional_json(manifest_path)
    assert summary is not None and manifest is not None
    declared: dict[str, str] = {}
    for record in manifest.get("outputs", []):
        if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
            raise IntegrityError("synthetic manifest contains an invalid output record")
        declared[Path(str(record["path"])).name] = str(record["sha256"])
    required_hashed = expected - {"manifest.json"}
    if set(declared) != required_hashed:
        raise IntegrityError("synthetic manifest does not bind every persisted source artifact")
    sources = [directory / name for name in sorted(expected)]
    for path in sources:
        if path.name != "manifest.json" and sha256_file(path) != declared[path.name]:
            raise IntegrityError(f"synthetic source hash mismatch: {path.name}")
    counts = manifest.get("row_counts")
    if isinstance(counts, Mapping):
        count_paths = {
            "exact_indistinguishable": directory / "exact_indistinguishable_rows.jsonl",
            "approximate_two_world": directory / "approximate_two_world_rows.jsonl",
            "identifiable_nongaussian": directory / "identifiable_nongaussian_rows.jsonl",
        }
        for key, path in count_paths.items():
            if int(counts.get(key, -1)) != len(_read_jsonl(path)):
                raise IntegrityError(f"synthetic row count mismatch for {key}")
    return summary, sources


def _load_controlled_sources(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    """Load a complete controlled suite or report it absent; validate its bindings."""

    directory = root / "controlled"
    expected = {"manifest.json", "pair_rank_phase.jsonl", "real_embedding_orbits.json"}
    present = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    if not (present & expected):
        return [], [], []
    missing = expected - present
    if missing:
        raise IntegrityError(f"partial controlled suite; missing {sorted(missing)}")
    manifest_path = directory / "manifest.json"
    pair_path = directory / "pair_rank_phase.jsonl"
    orbit_path = directory / "real_embedding_orbits.json"
    manifest = _load_optional_json(manifest_path)
    orbit = _load_optional_json(orbit_path)
    assert manifest is not None and orbit is not None
    if manifest.get("pair_rows_sha256") != sha256_file(pair_path):
        raise IntegrityError("controlled pair rows do not match their manifest")
    if manifest.get("orbit_sha256") != sha256_file(orbit_path):
        raise IntegrityError("controlled orbit collection does not match its manifest")
    pair_rows = _read_jsonl(pair_path)
    orbit_rows = orbit.get("records")
    if not isinstance(orbit_rows, list):
        raise IntegrityError("controlled orbit collection has no records list")
    if int(manifest.get("pair_rows", -1)) != len(pair_rows):
        raise IntegrityError("controlled pair row count does not match its manifest")
    if int(manifest.get("orbit_records", -1)) != len(orbit_rows):
        raise IntegrityError("controlled orbit count does not match its manifest")
    return pair_rows, list(orbit_rows), [manifest_path, pair_path, orbit_path]


def _load_streaming_sources(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    directory = root / "streaming"
    manifest_path = directory / "manifest.json"
    rows_path = directory / "aggregate_rows.jsonl"
    if not manifest_path.is_file() and not rows_path.is_file():
        return [], []
    if not manifest_path.is_file() or not rows_path.is_file():
        raise IntegrityError("partial streaming collection")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise IntegrityError("streaming collection is not PASS")
    if manifest.get("protocol") != "strict_predict_before_update":
        raise IntegrityError("streaming collection is not strict prequential")
    if manifest.get("aggregate_rows_sha256") != sha256_file(rows_path):
        raise IntegrityError("streaming aggregates do not match their manifest")
    rows = _read_jsonl(rows_path)
    if len(rows) != int(manifest.get("aggregate_rows", -1)):
        raise IntegrityError("streaming aggregate row count differs from manifest")
    if int(manifest.get("condition_units", -1)) != int(
        manifest.get("expected_condition_units", -2)
    ):
        raise IntegrityError("streaming condition grid is incomplete")
    return rows, [manifest_path, rows_path]


def generate_paper_outputs(run_dir: str | Path) -> list[Path]:
    root = Path(run_dir).expanduser().resolve()
    output = root / "paper_outputs"
    output.mkdir(parents=True, exist_ok=True)
    # This directory is generated-only. Remove old files so a rerun cannot leave
    # unmanifested figures from an optional panel that is no longer available.
    for path in output.iterdir():
        if path.is_file():
            path.unlink()

    raw_rows, summary_groups, real_sources = _verify_real_inputs(root)
    generated: list[Path] = []
    panel_status: dict[str, str] = {}

    common_real_columns: list[Column] = [
        ("dataset", "Dataset", lambda row: str(row.get("dataset", "--"))),
        ("encoder", "Encoder", lambda row: str(row.get("encoder", "--"))),
        ("method", "Method", lambda row: _label(row.get("method"))),
        ("agreement", "Agreement (%)", lambda row: _format_pm(row, "reference_prediction_agreement", percent=True)),
        ("accuracy", "Accuracy (%)", lambda row: _format_pm(row, "aligned_accuracy", percent=True)),
        ("covariance", "Cov. residual", lambda row: _format_pm(row, "aligned_covariance_relative_error")),
        ("sliced", "Sliced W2²", lambda row: _format_pm(row, "aligned_sliced_w2_squared")),
        ("dependence", "Source corr. change", lambda row: _format_pm(
            row, "source_to_aligned_correlation_relative_change"
        )),
    ]

    primary = _sorted_rows([
        row for row in summary_groups if row.get("suite") == "primary_marginal"
        and int(row.get("calibration_size", -1)) == PRIMARY_CALIBRATION_SIZE
    ], ["dataset", "encoder", "method"])
    if not primary:
        raise IntegrityError(
            f"locked primary_marginal endpoint at n={PRIMARY_CALIBRATION_SIZE} is absent"
        )
    paths = _write_panel(
        output, name="main_marginal_ladder", title="Main marginal information ladder",
        rows=primary, columns=common_real_columns, sources=real_sources, root=root,
        notes=(
            f"Locked primary calibration size: n={PRIMARY_CALIBRATION_SIZE} per endpoint.",
            "Only methods present in persisted primary-marginal rows are reported.",
            "Distribution residuals and frozen-task outcomes are shown together.",
            "Source-correlation change audits the diagonal theorem separately from reference matching.",
        ),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_main(primary, output))

    primary_raw = [row for row in raw_rows if row.get("suite") == "severity_breadth"
                   and int(row.get("calibration_size", -1)) == PRIMARY_CALIBRATION_SIZE]
    breadth = _macro_aggregate(
        primary_raw, keys=("dataset", "encoder", "severity", "method"),
        metrics=("reference_prediction_agreement", "aligned_accuracy",
                 "aligned_covariance_relative_error"),
    ) if primary_raw else []
    paths = _write_panel(
        output, name="encoder_severity_breadth", title="Encoder and severity breadth",
        rows=breadth,
        columns=[
            *common_real_columns[:3],
            ("severity", "Severity", lambda row: str(row.get("severity"))),
            *common_real_columns[3:6],
        ], sources=real_sources, root=root,
        notes=(
            "This secondary suite covers severities 1/3/5; the locked primary endpoint uses severity 5.",
            "Corruptions are macro-averaged within each split; SD is across split repeats.",
        ),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_method_lines(
        breadth, output, "encoder_severity_breadth", x_key="severity",
        y_metric="reference_prediction_agreement", xlabel="Corruption severity",
        ylabel="Frozen-reference prediction agreement (%)",
    ))

    suite_specs = (
        ("finite_sample", "Finite-sample estimation", "finite_sample", "calibration_size",
         "Unpaired calibration samples per endpoint", True),
        ("rank_shrinkage", "Rank and shrinkage ablation", "rank_shrinkage", "calibration_size",
         "Calibration samples per endpoint", True),
        ("higher_order", "Higher-order marginal panel", "higher_order_representative",
         "calibration_size", "Calibration samples per endpoint", True),
    )
    for name, title, suite, x_key, xlabel, log_x in suite_specs:
        selected = _sorted_rows([row for row in summary_groups if row.get("suite") == suite],
                                ["dataset", "encoder", "method", "calibration_size"])
        paths = _write_panel(
            output, name=name, title=title, rows=selected,
            columns=[*common_real_columns[:3],
                     ("n", "n", lambda row: str(row.get("calibration_size"))),
                     *common_real_columns[3:]],
            sources=real_sources, root=root,
            notes=("Every displayed method and budget is discovered from persisted rows.",),
        )
        generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
        generated.extend(_plot_method_lines(
            selected, output, name, x_key=x_key, y_metric="reference_prediction_agreement",
            xlabel=xlabel, ylabel="Frozen-reference prediction agreement (%)", log_x=log_x,
        ))

    correspondence_raw = [row for row in raw_rows if row.get("suite") == "correspondence_dose"
                          and row.get("pair_budget") is not None]
    for row in correspondence_raw:
        method = str(row.get("method", ""))
        row["estimator_family"] = re.sub(r"_shuffled$|_noise[-+0-9.eE]+$", "", method)
    correspondence = _macro_aggregate(
        correspondence_raw,
        keys=("dataset", "encoder", "estimator_family", "method", "pair_control",
              "pair_budget", "pair_noise_rate_requested", "pair_noise_rate_actual"),
        metrics=("reference_prediction_agreement", "aligned_accuracy", "paired_feature_mse",
                 "centered_logit_mse", "aligned_covariance_relative_error"),
    ) if correspondence_raw else []
    correspondence_columns: list[Column] = [
        ("dataset", "Dataset", lambda row: str(row.get("dataset"))),
        ("encoder", "Encoder", lambda row: str(row.get("encoder"))),
        ("method", "Estimator", lambda row: _label(row.get("method"))),
        ("control", "Pair map", lambda row: str(row.get("pair_control"))),
        ("budget", "Pairs", lambda row: str(row.get("pair_budget"))),
        ("requested", "Requested noise", lambda row: _format_value(row.get("pair_noise_rate_requested"))),
        ("actual", "Actual noise", lambda row: _format_value(row.get("pair_noise_rate_actual"))),
        ("agreement", "Agreement (%)", lambda row: _format_pm(row, "reference_prediction_agreement", percent=True)),
        ("accuracy", "Accuracy (%)", lambda row: _format_pm(row, "aligned_accuracy", percent=True)),
        ("feature", "Paired MSE", lambda row: _format_pm(row, "paired_feature_mse")),
    ]
    paths = _write_panel(
        output, name="correspondence_dose", title="Correspondence dose and matched controls",
        rows=correspondence, columns=correspondence_columns, sources=real_sources, root=root,
        notes=(
            "True and shuffled rows retain estimator identity and pair budget.",
            "Requested and actually realized mismatch fractions are reported separately.",
            "These finite-sample estimators are not labeled theoretical ceilings.",
        ),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_method_lines(
        correspondence, output, "correspondence_dose", x_key="pair_budget",
        y_metric="reference_prediction_agreement", xlabel="Paired calibration examples",
        ylabel="Frozen-reference prediction agreement (%)",
        group_keys=("dataset", "encoder", "method", "pair_noise_rate_actual"), log_x=True,
    ))

    er_raw = []
    for raw in raw_rows:
        if raw.get("method") != "labeled_er" and not bool(raw.get("changes_frozen_consumer")):
            continue
        row = dict(raw)
        diagnostics = row.get("diagnostics", {})
        row["target_labeled_rows"] = diagnostics.get("target_buffer_rows")
        row["clean_replay_rows"] = diagnostics.get("clean_reference_replay_rows")
        er_raw.append(row)
    er = _macro_aggregate(
        er_raw, keys=("dataset", "encoder", "target_labeled_rows", "clean_replay_rows", "method"),
        metrics=("reference_prediction_agreement", "aligned_accuracy", "negative_log_likelihood",
                 "brier_score", "expected_calibration_error",
                 "aligned_covariance_relative_error", "aligned_sliced_w2_squared"),
    ) if er_raw else []
    paths = _write_panel(
        output, name="labeled_er", title="Labeled experience-replay comparator",
        rows=er,
        columns=[*common_real_columns[:3],
                 ("target_n", "Target labels", lambda row: str(row.get("target_labeled_rows"))),
                 ("replay_n", "Clean replay", lambda row: str(row.get("clean_replay_rows"))),
                 *common_real_columns[3:],
                 ("nll", "NLL", lambda row: _format_pm(row, "negative_log_likelihood")),
                 ("ece", "ECE", lambda row: _format_pm(row, "expected_calibration_error"))],
        sources=real_sources, root=root,
        notes=("This comparator changes the consumer and is not a marginal-only feature map.",),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_method_lines(
        er, output, "labeled_er", x_key="target_labeled_rows",
        y_metric="aligned_accuracy", xlabel="Labeled target examples",
        ylabel="Accuracy (%)", group_keys=("dataset", "encoder", "method"), log_x=True,
    ))

    streaming_rows, streaming_sources = _load_streaming_sources(root)
    streaming = _macro_aggregate(
        streaming_rows,
        keys=("dataset", "encoder", "strategy", "method", "history_mode", "window_size"),
        metrics=("reference_prediction_agreement_prequential_mean",
                 "aligned_accuracy_prequential_mean",
                 "aligned_covariance_relative_error_prequential_mean"),
        repeat_key="order_seed",
    ) if streaming_rows else []
    paths = _write_panel(
        output, name="strict_prequential_streaming",
        title="Strict prequential streaming audit",
        rows=streaming,
        columns=[
            ("dataset", "Dataset", lambda row: str(row.get("dataset"))),
            ("encoder", "Encoder", lambda row: str(row.get("encoder"))),
            ("strategy", "Strategy", lambda row: str(row.get("strategy"))),
            ("history", "History", lambda row: str(row.get("history_mode"))),
            ("window", "Window", lambda row: "--" if row.get("window_size") is None else str(row.get("window_size"))),
            ("agreement", "Prequential agreement (%)", lambda row: _format_pm(
                row, "reference_prediction_agreement_prequential_mean", percent=True
            )),
            ("accuracy", "Prequential accuracy (%)", lambda row: _format_pm(
                row, "aligned_accuracy_prequential_mean", percent=True
            )),
            ("covariance", "Prequential cov. residual", lambda row: _format_pm(
                row, "aligned_covariance_relative_error_prequential_mean"
            )),
        ], sources=streaming_sources, root=root,
        notes=(
            "Every block is predicted before its target features update cumulative/window statistics.",
            "This is a secondary deployment audit, distinct from the endpoint-marginal primary analysis.",
        ),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]

    inference_path = root / "inference.json"
    inference = _load_optional_json(inference_path)
    inference_rows = list(inference.get("contrasts", [])) if inference else []
    inference_sources = [inference_path] if inference else []
    paths = _write_panel(
        output, name="primary_inference", title="Locked primary inference",
        rows=inference_rows,
        columns=[
            ("dataset", "Dataset", lambda row: str(row.get("dataset", "all"))),
            ("contrast", "Contrast", lambda row: f"{_label(row.get('left'))} − {_label(row.get('right'))}"),
            ("metric", "Metric", lambda row: str(row.get("metric"))),
            ("effect", "Effect", lambda row: _format_value(row.get("effect"), digits=5)),
            ("ci", "95% CI", lambda row: f"[{_format_value(row.get('ci_lower'), digits=5)}, {_format_value(row.get('ci_upper'), digits=5)}]"),
            ("raw", "raw p", lambda row: _format_value(row.get("raw_p"), digits=5)),
            ("holm", "Holm p", lambda row: _format_value(row.get("holm_p"), digits=5)),
            ("clusters", "Clusters", lambda row: str(row.get("clusters"))),
        ], sources=inference_sources, root=root,
        notes=((inference or {}).get("pooling_policy", "No inference artifact was available."),),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_inference(inference_rows, output))

    synthetic, synthetic_sources = _load_synthetic_sources(root)
    synthetic_manifest = bool(synthetic_sources)
    exact_raw_path = root / "synthetic/exact_indistinguishable_rows.jsonl"
    exact_raw = _read_jsonl(exact_raw_path)
    exact = _macro_aggregate(
        exact_raw, keys=("family", "angle_degrees", "classes"),
        metrics=("theoretical_equal_prior_bayes_accuracy", "empirical_equal_prior_bayes_accuracy",
                 "uniform_inverse_world_gap", "absolute_error"),
        repeat_key="seed",
    ) if exact_raw else []
    paths = _write_panel(
        output, name="synthetic_exact", title="Exact indistinguishable-world ceilings",
        rows=exact,
        columns=[
            ("family", "Family", lambda row: str(row.get("family"))),
            ("angle", "Angle", lambda row: "--" if row.get("angle_degrees") is None else str(row.get("angle_degrees"))),
            ("classes", "Classes/worlds", lambda row: "--" if row.get("classes") is None else str(row.get("classes"))),
            ("theory", "Exact", lambda row: _format_pm(row, "theoretical_equal_prior_bayes_accuracy", percent=True)),
            ("mc", "Monte Carlo", lambda row: _format_pm(row, "empirical_equal_prior_bayes_accuracy", percent=True)),
            ("error", "Abs. error", lambda row: _format_pm(row, "absolute_error")),
        ], sources=synthetic_sources, root=root,
        notes=("Numerical ceilings apply only to the explicitly declared indistinguishable worlds.",),
        status="PASS" if exact and synthetic_manifest else None,
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_synthetic_exact(exact_raw, output))

    approximate = list((synthetic or {}).get("approximate_two_world", {}).get("groups", []))
    paths = _write_panel(
        output, name="synthetic_approximate", title="Approximate two-world ceiling",
        rows=approximate,
        columns=[
            ("n", "Calibration n", lambda row: str(row.get("calibration_samples"))),
            ("epsilon", "ε", lambda row: _format_value(row.get("epsilon"))),
            ("delta", "Task disagreement", lambda row: _format_value(row.get("task_disagreement"))),
            ("tv", "Joint TV", lambda row: _format_pm(row, "joint_observation_total_variation")),
            ("exact", "Exact ceiling (%)", lambda row: _format_pm(row, "exact_equal_prior_bayes_accuracy", percent=True)),
            ("mc", "Monte Carlo (%)", lambda row: _format_pm(row, "empirical_bayes_accuracy", percent=True)),
        ], sources=synthetic_sources, root=root,
        notes=("The complete calibration observation law, not one-sample TV, determines this panel.",),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_synthetic_approximate(approximate, output))

    nongaussian = list((synthetic or {}).get("identifiable_nongaussian", {}).get("groups", []))
    paths = _write_panel(
        output, name="synthetic_nongaussian", title="Positive non-Gaussian identifiability",
        rows=nongaussian,
        columns=[
            ("n", "Calibration n", lambda row: str(row.get("calibration_samples"))),
            ("asymmetry", "Asymmetry", lambda row: _format_value(row.get("asymmetry"))),
            ("method", "Method", lambda row: _label(row.get("method"))),
            ("orientation", "Orientation correct (%)", lambda row: _format_pm(row, "orientation_correct", percent=True)),
            ("task", "Task agreement (%)", lambda row: _format_pm(row, "task_agreement", percent=True)),
            ("covariance", "Cov. residual", lambda row: _format_pm(row, "heldout_covariance_relative_error")),
        ], sources=synthetic_sources, root=root,
        notes=("This positive control shows when higher-order marginals can identify orientation.",),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_method_lines(
        nongaussian, output, "synthetic_nongaussian", x_key="calibration_samples",
        y_metric="orientation_correct", xlabel="Calibration observations",
        ylabel="Orientation recovery (%)", group_keys=("asymmetry", "method"), log_x=True,
    ))

    pair_rows, orbit_rows, controlled_sources = _load_controlled_sources(root)
    pair_summary = _macro_aggregate(
        pair_rows, keys=("dimension", "classes", "method", "pair_budget",
                         "pairs_per_dimension", "pair_noise_rate"),
        metrics=("prediction_agreement", "paired_feature_mse", "centered_logit_mse",
                 "identified_centered_readout_energy"),
        repeat_key="seed",
    ) if pair_rows else []
    paths = _write_panel(
        output, name="controlled_pair_phase", title="Controlled pair-rank phase diagram",
        rows=pair_summary,
        columns=[
            ("dimension", "d", lambda row: str(row.get("dimension"))),
            ("classes", "Classes", lambda row: str(row.get("classes"))),
            ("method", "Estimator", lambda row: _label(row.get("method"))),
            ("pairs", "Pairs", lambda row: str(row.get("pair_budget"))),
            ("ratio", "Pairs/d", lambda row: _format_value(row.get("pairs_per_dimension"))),
            ("noise", "Noise", lambda row: _format_value(row.get("pair_noise_rate"))),
            ("agreement", "Agreement (%)", lambda row: _format_pm(row, "prediction_agreement", percent=True)),
            ("feature", "Paired MSE", lambda row: _format_pm(row, "paired_feature_mse")),
            ("energy", "Identified task energy", lambda row: _format_pm(row, "identified_centered_readout_energy")),
        ], sources=controlled_sources, root=root,
        notes=("Threshold budgets around d are retained; pair rank and task-relevant rank are distinct.",),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_method_lines(
        pair_summary, output, "controlled_pair_phase", x_key="pairs_per_dimension",
        y_metric="prediction_agreement", xlabel="Pairs per feature dimension",
        ylabel="Prediction agreement (%)", group_keys=("dimension", "classes", "method", "pair_noise_rate"),
        log_x=True,
    ))

    paths = _write_panel(
        output, name="controlled_real_orbit", title="Controlled real-embedding group orbit",
        rows=orbit_rows,
        columns=[
            ("dataset", "Dataset", lambda row: str(row.get("dataset"))),
            ("encoder", "Encoder", lambda row: str(row.get("encoder"))),
            ("worlds", "Worlds", lambda row: str(row.get("worlds"))),
            ("rank", "PCA rank", lambda row: str(row.get("rank"))),
            ("bayes", "Unrestricted Bayes upper (%)", lambda row: _format_value(row.get("unrestricted_equal_prior_bayes_prediction_upper"), percent=True)),
            ("minimax", "Group-map minimax (%)", lambda row: _format_value(row.get("achievable_group_map_randomized_minimax"), percent=True)),
            ("mean", "Mean residual", lambda row: _format_value(row.get("endpoint_mean_residual"))),
            ("cov", "Cov. residual", lambda row: _format_value(row.get("endpoint_covariance_relative_residual"))),
        ], sources=controlled_sources, root=root,
        notes=("Unrestricted Bayes and achievable randomized group-map minimax are distinct objects.",),
    )
    generated.extend(paths[:2]); panel_status[paths[2]["name"]] = paths[2]["status"]
    generated.extend(_plot_orbits(orbit_rows, output))

    # Every generated file except the manifest is inventoried with exact bytes/hash.
    generated = sorted({path.resolve() for path in generated}, key=lambda path: path.name)
    manifest_path = output / "manifest.json"
    write_json_atomic(manifest_path, {
        "schema_version": "marginal_alignment.campaign.paper_outputs.v2",
        "status": "PASS",
        "primary_calibration_size": PRIMARY_CALIBRATION_SIZE,
        "panel_status": panel_status,
        "source_artifacts": [_source_record(root, path) for path in sorted(
            {path.resolve() for path in [*real_sources, *inference_sources, *synthetic_sources,
                                         *controlled_sources, *streaming_sources] if path.is_file()}, key=str
        )],
        "outputs": [
            {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
             "sha256": sha256_file(path)} for path in generated
        ],
    })
    return [*generated, manifest_path]
