#!/usr/bin/env python3
"""Evaluator interface for a scripted velocity-controller baseline.

The repository's main policy is a learned low-level joint-position controller.
This baseline is reserved for a simple hand-written command policy such as
``v_x = const`` and ``omega_z = k_yaw * yaw_error + k_y * lateral_error``. A
full Isaac Sim rollout implementation is intentionally not faked here: use
``--write_template`` to create the required CSV schema, then replace it with
real rollout rows when the scripted controller is connected to the simulator.
"""

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


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/narrow_passage_eval/scripted_velocity_controller.csv"))
    parser.add_argument("--write_template", action="store_true", help="Write a header-only CSV with the shared schema.")
    args = parser.parse_args()

    if args.write_template:
        write_template(args.output)
        print(f"Wrote scripted-controller CSV template to {args.output}.")
        return

    raise SystemExit(
        "Scripted velocity-controller rollout is an evaluator interface only. "
        "Run with --write_template for the required CSV schema, or connect this "
        "script to an Isaac Sim rollout before reporting baseline numbers."
    )


if __name__ == "__main__":
    main()
