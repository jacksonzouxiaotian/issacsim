#!/usr/bin/env python3
"""Build compact markdown tables from narrow-passage evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = (
    "success",
    "raw_success",
    "collision",
    "wedge",
    "rejected",
    "oscillation_count",
    "completion_time",
    "min_clearance",
)


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["_file"] = path.name
                rows.append(row)
    return rows


def summarize_group(rows: list[dict[str, str]]) -> dict[str, float]:
    summary = {}
    for metric in METRICS:
        values = [_float(row, metric, math.nan) for row in rows if row.get(metric, "") != ""]
        summary[metric] = _mean(values) if values else math.nan
    return summary


def group_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def width_scan_table(rows: list[dict[str, str]]) -> str:
    body = []
    groups = group_rows(rows, ("controller", "scenario", "zero_recovery_memory", "width"))
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2], float(item[3] or 0.0))):
        controller, scenario, zero_memory, width = key
        summary = summarize_group(groups[key])
        body.append(
            [
                controller,
                scenario or "nominal",
                zero_memory or "0",
                width,
                _fmt(summary["success"]),
                _fmt(summary["raw_success"]),
                _fmt(summary["collision"]),
                _fmt(summary["wedge"]),
                _fmt(summary["rejected"]),
                _fmt(summary["completion_time"]),
                _fmt(summary["min_clearance"]),
                _fmt(summary["oscillation_count"], 2),
            ]
        )
    return markdown_table(
        [
            "controller",
            "scenario",
            "zero_memory",
            "width",
            "clean_SR",
            "raw_SR",
            "collision",
            "wedge",
            "reject",
            "time",
            "min_clearance",
            "osc",
        ],
        body,
    )


def delta_d_table(rows: list[dict[str, str]]) -> str:
    body = []
    groups = group_rows(rows, ("controller", "scenario", "delta_d"))
    for key in sorted(groups, key=lambda item: (item[0], item[1], float(item[2] or 0.0))):
        controller, scenario, delta_d = key
        summary = summarize_group(groups[key])
        body.append(
            [
                controller,
                scenario or "nominal",
                delta_d,
                _fmt(summary["success"]),
                _fmt(summary["wedge"]),
                _fmt(summary["rejected"]),
                _fmt(1.0 - summary["success"]),
            ]
        )
    return markdown_table(
        ["controller", "scenario", "delta_D", "success_prob", "wedge_prob", "reject_prob", "calibration_error"],
        body,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("logs/narrow_passage_eval"))
    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument("--output", type=Path, default=Path("logs/narrow_passage_eval/eval_tables.md"))
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No CSV files found under {args.input_dir} with pattern {args.pattern!r}")

    rows = read_rows(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(
        [
            "# Narrow-Passage Evaluation Tables",
            "## Width, Recovery, and Generalization Summary",
            width_scan_table(rows),
            "## Delta-D Calibration Inputs",
            delta_d_table(rows),
        ]
    )
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {args.output} from {len(paths)} CSV files and {len(rows)} trials.")


if __name__ == "__main__":
    main()
