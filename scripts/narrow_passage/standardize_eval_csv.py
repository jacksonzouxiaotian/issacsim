#!/usr/bin/env python3
"""Convert narrow-passage evaluation CSV files to the shared ablation schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = [
    "method",
    "scenario",
    "width",
    "trial",
    "success",
    "collision",
    "wedge",
    "rejected",
    "oscillation_count",
    "completion_time",
    "min_clearance",
]

ALIASES = {
    "method": ("method", "controller", "policy", "task"),
    "scenario": ("scenario",),
    "width": ("width",),
    "trial": ("trial", "episode", "episode_id", "env_id"),
    "success": ("success", "clean_success"),
    "collision": ("collision",),
    "wedge": ("wedge",),
    "rejected": ("rejected", "reject"),
    "oscillation_count": ("oscillation_count",),
    "completion_time": ("completion_time", "time_to_goal"),
    "min_clearance": ("min_clearance",),
}


def first_value(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return default


def normalize_row(row: dict[str, str], method: str, default_scenario: str) -> dict[str, str]:
    normalized = {}
    for field in OUTPUT_FIELDS:
        default = "0" if field in {"collision", "wedge", "rejected", "oscillation_count"} else ""
        normalized[field] = first_value(row, ALIASES[field], default=default)
    normalized["method"] = method or normalized["method"]
    normalized["scenario"] = normalized["scenario"] or default_scenario
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing evaluator CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Standardized output CSV.")
    parser.add_argument("--method", type=str, required=True, help="Ablation/baseline method name.")
    parser.add_argument("--default_scenario", type=str, default="nominal")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="", encoding="utf-8") as in_stream:
        rows = list(csv.DictReader(in_stream))

    with args.output.open("w", newline="", encoding="utf-8") as out_stream:
        writer = csv.DictWriter(out_stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row(row, args.method, args.default_scenario))

    print(f"Wrote {args.output} with {len(rows)} standardized rows for method={args.method}.")


if __name__ == "__main__":
    main()
