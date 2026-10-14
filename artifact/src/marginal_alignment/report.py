"""Render auditable paper tables from a persisted summary, never from logs."""

from __future__ import annotations

import json
from pathlib import Path


DISPLAY = {
    "none": "No alignment",
    "mean": "Mean",
    "diagonal": "Diagonal / BN-style",
    "coral": "CORAL",
    "bures": "Bures",
    "bures_rank4": "Bures rank 4",
    "bures_rank64": "Bures rank 64",
    "paired_procrustes": "Paired Procrustes",
    "paired_oracle": "Paired covariance oracle",
    "paired_readout_oracle": "Paired readout oracle",
    "shuffled_pairs": "Shuffled pairs",
    "er": "ER (changes readout)",
}


def render_real_table(
    summary_path: str | Path,
    output_dir: str | Path,
    *,
    calibration_size: int = 1024,
) -> list[Path]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    rows = [
        row for row in summary["groups"]
        if int(row["calibration_size"]) == int(calibration_size)
    ]
    if not rows:
        raise ValueError(f"no summary rows at calibration size {calibration_size}")
    rows.sort(key=lambda row: (
        str(row["encoder"]),
        list(DISPLAY).index(row["method"]) if row["method"] in DISPLAY else 999,
    ))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    markdown = [
        "| Encoder | Information | Accuracy | Ref.-prediction agreement | Covariance residual |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['encoder']} | {DISPLAY.get(row['method'], row['method'])} "
            f"| {100*row['aligned_accuracy_mean']:.2f} ± {100*row['aligned_accuracy_std']:.2f} "
            f"| {100*row['reference_prediction_agreement_mean']:.2f} ± "
            f"{100*row['reference_prediction_agreement_std']:.2f} "
            f"| {row['aligned_covariance_relative_error_mean']:.3f} ± "
            f"{row['aligned_covariance_relative_error_std']:.3f} |"
        )
    md_path = output / "real_main_table.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Encoder & Information & Accuracy (\%) & Ref. agree. (\%) & Cov. residual \\",
        r"\midrule",
    ]
    for row in rows:
        name = DISPLAY.get(row["method"], row["method"]).replace("_", r"\_")
        encoder = str(row["encoder"]).replace("_", r"\_")
        latex.append(
            f"{encoder} & {name}"
            f" & {100*row['aligned_accuracy_mean']:.2f}"
            f" $\\pm$ {100*row['aligned_accuracy_std']:.2f}"
            f" & {100*row['reference_prediction_agreement_mean']:.2f}"
            f" $\\pm$ {100*row['reference_prediction_agreement_std']:.2f}"
            f" & {row['aligned_covariance_relative_error_mean']:.3f}"
            f" $\\pm$ {row['aligned_covariance_relative_error_std']:.3f} \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}"]
    tex_path = output / "real_main_table.tex"
    tex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")

    provenance = {
        "summary_source": str(Path(summary_path).resolve()),
        "calibration_size": calibration_size,
        "n_rows": len(rows),
        "methods": [row["method"] for row in rows],
        "note": (
            "Mean and standard deviation are over persisted corruption × split-seed rows. "
            "ER uses target labels and changes a copy of the frozen consumer; all alignment "
            "rows leave the consumer fixed."
        ),
    }
    provenance_path = output / "real_main_table_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return [md_path, tex_path, provenance_path]
