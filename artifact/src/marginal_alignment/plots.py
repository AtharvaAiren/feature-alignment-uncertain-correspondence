"""Paper-facing plots generated only from persisted result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


COLORS = {
    "none": "#777777",
    "mean": "#56B4E9",
    "diagonal": "#0072B2",
    "coral": "#009E73",
    "bures": "#E69F00",
    "bures_rank4": "#D55E00",
    "bures_rank64": "#CC79A7",
    "paired_procrustes": "#6A3D9A",
    "paired_oracle": "#000000",
    "shuffled_pairs": "#BBBBBB",
    "er": "#A65628",
}


def _matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    })
    return plt


def _save(fig, output: Path, stem: str) -> list[Path]:
    outputs: list[Path] = []
    for suffix in ("pdf", "png"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None)
        outputs.append(path)
    return outputs


def _information_ladder(results: Path, output: Path) -> list[Path]:
    plt = _matplotlib()
    data = json.loads((results / "summary.json").read_text())["groups"]
    rows = [row for row in data if row["calibration_size"] == max(x["calibration_size"] for x in data)]
    datasets = sorted({row["dataset"] for row in rows}, key=lambda x: int(x.rsplit("k", 1)[1]))
    methods = ["none", "mean", "diagonal", "coral", "bures", "paired_oracle", "shuffled_pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    width = 0.105
    x = np.arange(len(datasets), dtype=float)
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    for index, method in enumerate(methods):
        values = [
            lookup[(dataset, method)].get(
                "ambiguity_uniform_task_agreement_mean",
                lookup[(dataset, method)]["reference_prediction_agreement_mean"],
            )
            for dataset in datasets
        ]
        errors = [
            lookup[(dataset, method)].get("ambiguity_uniform_task_agreement_std", 0.0)
            for dataset in datasets
        ]
        positions = x + (index - (len(methods) - 1) / 2) * width
        axes[0].bar(
            positions, values, width=width, yerr=errors, capsize=1.5,
            label=method.replace("_", " "),
            color=COLORS.get(method), edgecolor="white", linewidth=0.4,
        )
        worst = [
            lookup[(dataset, method)].get("ambiguity_worst_task_agreement_mean", value)
            for dataset, value in zip(datasets, values)
        ]
        axes[0].scatter(
            positions, worst, marker="v", s=12, color="white",
            edgecolor=COLORS.get(method), linewidth=0.7, zorder=4,
        )
    ceilings = [lookup[(dataset, "none")]["theoretical_bayes_ceiling"] for dataset in datasets]
    axes[0].scatter(x, ceilings, marker="_", s=320, linewidths=2.0, color="#D62728", label="uniform-world Bayes ceiling")
    axes[0].set_xticks(x, [f"K={dataset.rsplit('k',1)[1]}" for dataset in datasets])
    axes[0].set_ylabel("Uniform-world agreement")
    axes[0].set_ylim(0, 1.06)
    axes[0].set_title("Bars: mean; triangles: fixed-method worst world")
    for method in methods:
        accuracy = [
            lookup[(dataset, method)].get(
                "ambiguity_uniform_task_agreement_mean",
                lookup[(dataset, method)]["reference_prediction_agreement_mean"],
            )
            for dataset in datasets
        ]
        covariance = [
            max(lookup[(dataset, method)]["aligned_covariance_relative_error_mean"], 1e-8)
            for dataset in datasets
        ]
        axes[1].scatter(
            covariance, accuracy, label=method.replace("_", " "), s=34,
            color=COLORS.get(method), alpha=0.9, edgecolor="white", linewidth=0.4,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Held-out covariance residual (log scale)")
    axes[1].set_ylabel("Frozen-task agreement")
    axes[1].set_title("Statistical repair ≠ task recovery")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    outputs = _save(fig, output, "synthetic_information_ladder")
    plt.close(fig)
    return outputs


def _ceiling_curve(ceiling_file: Path, output: Path) -> list[Path]:
    plt = _matplotlib()
    records = json.loads(ceiling_file.read_text())["records"]
    binary = [row for row in records if row["family"] == "binary_gaussian_rotation"]
    multi = [row for row in records if row["family"] == "multiclass_cyclic"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    angles = sorted({row["angle_degrees"] for row in binary})
    empirical = [np.mean([row["equalizer_min_world_accuracy"] for row in binary if row["angle_degrees"] == angle]) for angle in angles]
    empirical_std = [np.std([row["equalizer_min_world_accuracy"] for row in binary if row["angle_degrees"] == angle], ddof=1) for angle in angles]
    theoretical = [next(row["theoretical_ceiling"] for row in binary if row["angle_degrees"] == angle) for angle in angles]
    axes[0].plot(angles, theoretical, color="black", label=r"theory $1-\theta/(2\pi)$")
    axes[0].errorbar(angles, empirical, yerr=empirical_std, fmt="o", color="#0072B2", capsize=2, label="randomized equalizer (worst world)")
    axes[0].set(xlabel="Hidden readout rotation (degrees)", ylabel="Marginal-only two-world ceiling", ylim=(0.45, 1.03))
    axes[0].legend(frameon=False)
    classes = sorted({row["classes"] for row in multi})
    values = [np.mean([row["equalizer_min_world_accuracy"] for row in multi if row["classes"] == classes_]) for classes_ in classes]
    axes[1].plot(classes, [1 / value for value in classes], color="black", label=r"theory $1/K$")
    axes[1].scatter(classes, values, color="#009E73", zorder=3, label="randomized equalizer (worst world)")
    axes[1].set(xlabel="Transitive label orbit size $K$", ylabel="Marginal-only group ceiling", xticks=classes, ylim=(0.08, 0.55))
    axes[1].legend(frameon=False)
    fig.tight_layout()
    outputs = _save(fig, output, "theoretical_ceiling_validation")
    plt.close(fig)
    return outputs


def make_all_plots(results_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _information_ladder(results_dir / "synthetic_information_ladder", output_dir)
    outputs += _ceiling_curve(results_dir / "ceiling_validation.json", output_dir)
    return outputs
