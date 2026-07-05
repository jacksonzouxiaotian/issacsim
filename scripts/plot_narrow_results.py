#!/usr/bin/env python3
"""Plot paper-style narrow-passage evaluation figures from CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ALIASES = {
    "method": ("method", "controller", "policy", "task"),
    "scenario": ("scenario",),
    "width": ("width",),
    "success": ("success", "clean_success"),
    "collision": ("collision",),
    "min_clearance": ("min_clearance", "min_clearance_mean"),
    "oscillation_count": ("oscillation_count",),
    "completion_time": ("completion_time", "time_to_goal", "mean_time_to_goal"),
}

SUMMARY_ALIASES = {
    "success": ("clean_success_rate", "success_rate", "success"),
    "collision": ("collision_rate", "collision"),
    "min_clearance": ("min_clearance_mean", "min_clearance"),
    "oscillation_count": ("oscillation_count",),
}

RECOVERY_SCENARIOS = ["nominal", "left_wall", "right_wall", "yaw_left", "yaw_right"]


def first_value(row: dict[str, object], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            return str(value)
    return default


def to_float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def scenario_label(value: str) -> str:
    normalized = value.strip()
    return normalized.removesuffix("_start")


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else math.nan


def load_csv_rows(input_dir: Path, pattern: str, method_filter: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.glob(pattern)):
        if path.name.endswith("_summary.csv"):
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                method = first_value(row, ALIASES["method"], path.stem)
                if method_filter and method_filter not in method and method_filter not in path.stem:
                    continue
                rows.append(
                    {
                        "source": path.name,
                        "method": method,
                        "scenario": scenario_label(first_value(row, ALIASES["scenario"], "nominal")),
                        "width": first_value(row, ALIASES["width"], ""),
                        "success": first_value(row, ALIASES["success"], "0"),
                        "collision": first_value(row, ALIASES["collision"], "0"),
                        "min_clearance": first_value(row, ALIASES["min_clearance"], ""),
                        "oscillation_count": first_value(row, ALIASES["oscillation_count"], ""),
                        "completion_time": first_value(row, ALIASES["completion_time"], ""),
                        "weight": "1",
                    }
                )
    return rows


def load_summary_rows(input_dir: Path, method_filter: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("*_summary.json")):
        method = path.stem.removesuffix("_summary")
        if method_filter and method_filter not in method:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "source": path.name,
                    "method": method,
                    "scenario": scenario_label(first_value(entry, ALIASES["scenario"], "nominal")),
                    "width": first_value(entry, ALIASES["width"], ""),
                    "success": first_value(entry, SUMMARY_ALIASES["success"], "0"),
                    "collision": first_value(entry, SUMMARY_ALIASES["collision"], "0"),
                    "min_clearance": first_value(entry, SUMMARY_ALIASES["min_clearance"], ""),
                    "oscillation_count": first_value(entry, SUMMARY_ALIASES["oscillation_count"], ""),
                    "completion_time": first_value(entry, ALIASES["completion_time"], ""),
                    "weight": first_value(entry, ("num_trials",), "1"),
                }
            )
    return rows


def grouped_means(rows: list[dict[str, str]], keys: tuple[str, ...], metrics: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, float]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    summaries = {}
    for key, group in groups.items():
        summaries[key] = {metric: mean([to_float(row[metric]) for row in group]) for metric in metrics}
    return summaries


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_width_plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    nominal = [row for row in rows if row["scenario"] == "nominal" and row["width"]]
    data = grouped_means(nominal, ("width",), ("success", "collision"))
    widths = sorted(data, key=lambda key: to_float(key[0]))
    x = [to_float(width[0]) for width in widths]
    success = [data[width]["success"] for width in widths]
    collision = [data[width]["collision"] for width in widths]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(x, success, marker="o", linewidth=2.0, color="#1f77b4", label="Success rate")
    ax.plot(x, collision, marker="s", linewidth=2.0, color="#d62728", label="Collision rate")
    ax.set_xlabel("Passage width (m)")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "width_success_collision.png")
    plt.close(fig)


def save_recovery_plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    filtered = [row for row in rows if row["scenario"] in RECOVERY_SCENARIOS]
    data = grouped_means(filtered, ("scenario",), ("success", "collision"))
    scenarios = [scenario for scenario in RECOVERY_SCENARIOS if (scenario,) in data]
    x = range(len(scenarios))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar([i - width / 2 for i in x], [data[(s,)]["success"] for s in scenarios], width, color="#4c78a8", label="Success")
    ax.bar([i + width / 2 for i in x], [data[(s,)]["collision"] for s in scenarios], width, color="#e45756", label="Collision")
    ax.set_xticks(list(x), scenarios, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "scenario_recovery_failure.png")
    plt.close(fig)


def save_clearance_oscillation_plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    filtered = [row for row in rows if row["scenario"] in RECOVERY_SCENARIOS]
    data = grouped_means(filtered, ("scenario",), ("min_clearance", "oscillation_count"))
    scenarios = [scenario for scenario in RECOVERY_SCENARIOS if (scenario,) in data]
    x = range(len(scenarios))

    fig, ax1 = plt.subplots(figsize=(7.0, 3.8))
    ax2 = ax1.twinx()
    ax1.bar(x, [data[(s,)]["min_clearance"] for s in scenarios], width=0.55, color="#72b7b2", label="Min clearance")
    ax2.plot(list(x), [data[(s,)]["oscillation_count"] for s in scenarios], marker="o", linewidth=2.0, color="#f58518", label="Oscillation")
    ax1.set_xticks(list(x), scenarios, rotation=20, ha="right")
    ax1.set_ylabel("Mean min clearance (m)")
    ax2.set_ylabel("Mean oscillation count")
    ax1.grid(axis="y", color="0.85", linewidth=0.8)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    fig.savefig(output_dir / "clearance_oscillation.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("logs/narrow_passage_eval"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures"))
    parser.add_argument("--file_pattern", type=str, default="*.csv")
    parser.add_argument("--method", type=str, default=None, help="Optional substring filter for method or filename.")
    args = parser.parse_args()

    set_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(args.input_dir, args.file_pattern, args.method)
    source = "CSV"
    if not rows:
        rows = load_summary_rows(args.input_dir, args.method)
        source = "summary JSON"
    if not rows:
        raise SystemExit(f"No evaluation rows found in {args.input_dir}")

    save_width_plot(rows, args.output_dir)
    save_recovery_plot(rows, args.output_dir)
    save_clearance_oscillation_plot(rows, args.output_dir)
    print(f"Wrote figures to {args.output_dir} from {len(rows)} {source} rows.")


if __name__ == "__main__":
    main()
