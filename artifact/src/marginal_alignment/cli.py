"""Command-line entry points for generation, running, checking, and plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_config
from .synthetic import generate_cyclic_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marginal-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("generate-synthetic")
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--samples", type=int, default=12000)
    synthetic.add_argument("--dimension", type=int, default=16)
    synthetic.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    synthetic.add_argument("--classes", type=int, nargs="+", default=[2, 4, 8])
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", required=True)
    plot = subparsers.add_parser("plot")
    plot.add_argument("--results-dir", required=True)
    plot.add_argument("--output-dir", required=True)
    ceiling = subparsers.add_parser("validate-ceiling")
    ceiling.add_argument("--output", required=True)
    ceiling.add_argument("--samples", type=int, default=250000)
    ceiling.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    report = subparsers.add_parser("report")
    report.add_argument("--summary", required=True)
    report.add_argument("--output-dir", required=True)
    report.add_argument("--calibration-size", type=int, default=1024)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--rows", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--calibration-size", type=int, default=1024)
    analyze.add_argument("--bootstrap-replicates", type=int, default=20000)
    analyze.add_argument("--seed", type=int, default=20260811)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "generate-synthetic":
        paths = generate_cyclic_suite(
            arguments.output_dir, class_counts=tuple(arguments.classes),
            dimension=arguments.dimension, samples=arguments.samples,
            seeds=tuple(arguments.seeds),
        )
        print(json.dumps({"status": "PASS", "files": len(paths), "output": arguments.output_dir}))
        return 0
    if arguments.command == "run":
        result = run_config(arguments.config, arguments.output_dir)
        print(json.dumps(result["manifest"], indent=2, sort_keys=True))
        return 0
    if arguments.command == "plot":
        from .plots import make_all_plots

        outputs = make_all_plots(Path(arguments.results_dir), Path(arguments.output_dir))
        print(json.dumps({"status": "PASS", "outputs": [str(item) for item in outputs]}))
        return 0
    if arguments.command == "validate-ceiling":
        from .ceiling_experiment import run_ceiling_validation

        result = run_ceiling_validation(
            arguments.output, samples=arguments.samples, seeds=tuple(arguments.seeds)
        )
        print(json.dumps({
            "status": result["status"],
            "maximum_absolute_ceiling_error": result["maximum_absolute_ceiling_error"],
            "output": arguments.output,
        }))
        return 0 if result["status"] == "PASS" else 1
    if arguments.command == "report":
        from .report import render_real_table

        outputs = render_real_table(
            arguments.summary, arguments.output_dir,
            calibration_size=arguments.calibration_size,
        )
        print(json.dumps({"status": "PASS", "outputs": [str(item) for item in outputs]}))
        return 0
    if arguments.command == "analyze":
        from .statistics import analyze_real_rows

        result = analyze_real_rows(
            arguments.rows,
            arguments.output,
            calibration_size=arguments.calibration_size,
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed,
        )
        print(json.dumps({
            "status": "PASS", "records": len(result["records"]),
            "output": arguments.output,
        }))
        return 0
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    raise SystemExit(main())
