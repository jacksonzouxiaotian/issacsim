# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate narrow-passage local traversal policies.

This script is intentionally narrow in scope: it evaluates a local quadruped
passage policy in Isaac Sim and writes the metrics needed for width sweeps,
recovery ablations, and Delta-D calibration plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate narrow-passage local traversal policies.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Narrow-Anymal-C-v0")
parser.add_argument(
    "--controller",
    type=str,
    default="heuristic_rpp_like",
    choices=[
        "heuristic_dwb_like",
        "heuristic_rpp_like",
        "heuristic_mppi_like",
        "dwb",
        "rpp",
        "mppi",
        "tebrl",
        "checkpoint",
    ],
    help=(
        "Use checkpoint for learned policies. The *_like controllers are simple local heuristics, "
        "not real Nav2 DWB/RPP/MPPI implementations. Legacy aliases dwb/rpp/mppi are accepted."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL checkpoint for controller=checkpoint/tebrl.")
parser.add_argument("--widths", type=float, nargs="+", default=[0.75, 0.85, 0.95])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=220)
parser.add_argument("--estimated_d_min", type=float, default=0.72)
parser.add_argument("--safety_reject_margin", type=float, default=0.0)
parser.add_argument(
    "--recovery_scenario",
    type=str,
    default="nominal",
    choices=["nominal", "left_wall", "right_wall", "yaw_left", "yaw_right"],
)
parser.add_argument(
    "--zero_recovery_memory",
    action="store_true",
    default=False,
    help="Zero the last 5 recovery-memory observation channels before policy inference.",
)
parser.add_argument("--output", type=str, default="logs/narrow_passage_eval/metrics.csv")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


@dataclass
class EvalStats:
    min_clearance: torch.Tensor
    oscillations: torch.Tensor
    last_lat_sign: torch.Tensor
    completion_step: torch.Tensor
    done: torch.Tensor
    success: torch.Tensor
    collision: torch.Tensor
    wedge: torch.Tensor
    rejected: torch.Tensor


def _local_root_pos(env):
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, :3] - env.scene.env_origins


def _yaw_from_quat_wxyz(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _set_param_width(cfg_obj, width):
    for value in getattr(cfg_obj, "__dict__", {}).values():
        if hasattr(value, "params") and isinstance(value.params, dict) and "corridor_width" in value.params:
            value.params["corridor_width"] = width


def configure_straight_width(env_cfg, width):
    """Update straight-corridor scene geometry and all width-aware term params."""
    if not hasattr(env_cfg.scene, "left_wall"):
        return
    length = env_cfg.scene.left_wall.spawn.size[0]
    thickness = env_cfg.scene.left_wall.spawn.size[1]
    height = env_cfg.scene.left_wall.spawn.size[2]
    env_cfg.scene.left_wall.init_state.pos = (length / 2.0, -(width / 2.0 + thickness / 2.0), height / 2.0)
    env_cfg.scene.right_wall.init_state.pos = (length / 2.0, +(width / 2.0 + thickness / 2.0), height / 2.0)
    if hasattr(env_cfg.scene, "start_marker"):
        env_cfg.scene.start_marker.spawn.size = (0.03, width, 0.02)
    if hasattr(env_cfg.scene, "goal_marker"):
        env_cfg.scene.goal_marker.spawn.size = (0.03, width, 0.02)
    for cfg_obj in (env_cfg.observations.policy, env_cfg.rewards, env_cfg.terminations):
        _set_param_width(cfg_obj, width)


def configure_recovery_scenario(env_cfg, width, scenario):
    if scenario == "nominal":
        return
    velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    pose_range = env_cfg.events.reset_base.params.get("pose_range", {})
    if scenario == "left_wall":
        pose_range["x"] = (1.2, 1.8)
        pose_range["y"] = (width * 0.5 - 0.22, width * 0.5 - 0.16)
        pose_range["yaw"] = (-0.25, -0.12)
    elif scenario == "right_wall":
        pose_range["x"] = (1.2, 1.8)
        pose_range["y"] = (-(width * 0.5 - 0.22), -(width * 0.5 - 0.16))
        pose_range["yaw"] = (0.12, 0.25)
    elif scenario == "yaw_left":
        pose_range["x"] = (0.6, 1.2)
        pose_range["y"] = (-0.04, 0.04)
        pose_range["yaw"] = (0.35, 0.55)
    elif scenario == "yaw_right":
        pose_range["x"] = (0.6, 1.2)
        pose_range["y"] = (-0.04, 0.04)
        pose_range["yaw"] = (-0.55, -0.35)
    if "pose_range" in env_cfg.events.reset_base.params:
        env_cfg.events.reset_base.params["pose_range"] = pose_range
    elif "cases" in env_cfg.events.reset_base.params:
        env_cfg.events.reset_base.params["cases"] = (
            {"weight": 1.0, "pose_range": pose_range, "velocity_range": velocity_range},
        )


def heuristic_actions(env, controller, width):
    """Return local velocity commands for simple local baselines."""
    controller = controller.replace("heuristic_", "").replace("_like", "")
    robot = env.scene["robot"]
    pos = _local_root_pos(env)
    x = pos[:, 0]
    y = pos[:, 1]
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w)

    if controller == "dwb":
        v = torch.full_like(x, 0.32)
        lat = torch.clamp(-0.65 * y, -0.18, 0.18)
        wz = torch.clamp(-0.9 * yaw - 0.35 * y, -0.45, 0.45)
    elif controller == "mppi":
        candidates = [
            (0.25, -0.25),
            (0.35, -0.10),
            (0.45, 0.0),
            (0.35, 0.10),
            (0.25, 0.25),
        ]
        best_score = torch.full_like(x, -1.0e9)
        best_v = torch.zeros_like(x)
        best_lat = torch.zeros_like(x)
        for v_c, lat_c in candidates:
            next_y = y + 0.25 * lat_c
            clearance = width * 0.5 - torch.abs(next_y)
            score = 1.2 * v_c + 1.8 * clearance - 0.2 * torch.abs(next_y)
            update = score > best_score
            best_score[update] = score[update]
            best_v[update] = v_c
            best_lat[update] = lat_c
        v = best_v
        lat = best_lat
        wz = torch.clamp(-0.8 * yaw - 0.25 * y, -0.5, 0.5)
    else:
        v = torch.full_like(x, 0.48 if controller == "rpp" else 0.42)
        lat = torch.clamp(-0.85 * y, -0.25, 0.25)
        wz = torch.clamp(-1.1 * yaw - 0.45 * y, -0.55, 0.55)

    near_goal = x > 9.2
    v = torch.where(near_goal, torch.minimum(v, torch.full_like(v, 0.20)), v)
    return torch.stack([v, lat, wz], dim=-1)


def get_checkpoint_policy(wrapped_env, task, checkpoint):
    if checkpoint is None:
        raise ValueError("--checkpoint is required for controller=checkpoint/tebrl")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=wrapped_env.unwrapped.device)


def make_stats(num_envs, device, rejected):
    return EvalStats(
        min_clearance=torch.full((num_envs,), 1.0e9, device=device),
        oscillations=torch.zeros(num_envs, device=device),
        last_lat_sign=torch.zeros(num_envs, device=device),
        completion_step=torch.full((num_envs,), -1, dtype=torch.long, device=device),
        done=torch.zeros(num_envs, dtype=torch.bool, device=device),
        success=torch.zeros(num_envs, dtype=torch.bool, device=device),
        collision=torch.zeros(num_envs, dtype=torch.bool, device=device),
        wedge=torch.zeros(num_envs, dtype=torch.bool, device=device),
        rejected=torch.full((num_envs,), rejected, dtype=torch.bool, device=device),
    )


def update_stats(stats, env, width, step_idx):
    pos = _local_root_pos(env)
    robot = env.scene["robot"]
    clearance = width * 0.5 - torch.abs(pos[:, 1])
    stats.min_clearance = torch.minimum(stats.min_clearance, clearance)

    lat_sign = torch.sign(robot.data.root_lin_vel_w[:, 1])
    changed = (lat_sign != 0.0) & (stats.last_lat_sign != 0.0) & (lat_sign != stats.last_lat_sign)
    stats.oscillations += changed.float()
    stats.last_lat_sign[lat_sign != 0.0] = lat_sign[lat_sign != 0.0]

    speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    near_wall = clearance < 0.06
    stats.wedge |= near_wall & (speed < 0.035) & (env.episode_length_buf > 12)

    active = set(env.termination_manager.active_terms)
    if "goal_reached" in active:
        stats.success |= env.termination_manager.get_term("goal_reached")
    if "base_contact" in active:
        stats.collision |= env.termination_manager.get_term("base_contact")
    if "bad_orientation" in active:
        stats.collision |= env.termination_manager.get_term("bad_orientation")
    if "base_too_low" in active:
        stats.collision |= env.termination_manager.get_term("base_too_low")
    if "stuck" in active:
        stats.wedge |= env.termination_manager.get_term("stuck")

    just_done = env.reset_buf & (~stats.done)
    stats.done |= env.reset_buf
    stats.completion_step[just_done] = step_idx


def summarize(rows):
    by_width = {}
    for row in rows:
        width = row["width"]
        by_width.setdefault(width, []).append(row)
    summary = {}
    for width, items in by_width.items():
        n = max(len(items), 1)
        summary[f"SR@{int(round(width * 100))}"] = sum(i["success"] for i in items) / n
        summary[f"wedge@{int(round(width * 100))}"] = sum(i["wedge"] for i in items) / n
        summary[f"reject@{int(round(width * 100))}"] = sum(i["rejected"] for i in items) / n
        summary[f"collision@{int(round(width * 100))}"] = sum(i["collision"] for i in items) / n
    return summary


def run_width(width):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    configure_straight_width(env_cfg, width)
    configure_recovery_scenario(env_cfg, width, args_cli.recovery_scenario)
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    policy = None
    if args_cli.controller in ("checkpoint", "tebrl"):
        policy = get_checkpoint_policy(wrapped, args_cli.task, args_cli.checkpoint)

    rejected = width < (args_cli.estimated_d_min + args_cli.safety_reject_margin)
    stats = make_stats(wrapped.num_envs, wrapped.device, rejected)
    obs = wrapped.get_observations()
    if rejected:
        actions = torch.zeros(wrapped.num_envs, wrapped.num_actions, device=wrapped.device)
        stats.done[:] = True
        stats.completion_step[:] = 0
    else:
        for step_idx in range(args_cli.max_steps):
            with torch.inference_mode():
                if policy is None:
                    actions = heuristic_actions(wrapped.unwrapped, args_cli.controller, width)
                else:
                    if args_cli.zero_recovery_memory and "policy" in obs:
                        obs["policy"][:, -5:] = 0.0
                    actions = policy(obs)
                obs, _, _, _ = wrapped.step(actions)
            update_stats(stats, wrapped.unwrapped, width, step_idx + 1)
            if bool(stats.done.all()):
                break

    rows = []
    dt = wrapped.unwrapped.step_dt
    for env_id in range(wrapped.num_envs):
        step_count = int(stats.completion_step[env_id].item())
        if step_count < 0:
            step_count = args_cli.max_steps
        raw_success = bool(stats.success[env_id].item())
        collision = bool(stats.collision[env_id].item())
        wedge = bool(stats.wedge[env_id].item())
        rejected = bool(stats.rejected[env_id].item())
        clean_success = raw_success and not collision and not wedge and not rejected
        rows.append(
            {
                "controller": args_cli.controller,
                "scenario": args_cli.recovery_scenario,
                "zero_recovery_memory": int(args_cli.zero_recovery_memory),
                "width": float(width),
                "delta_d": float(width - args_cli.estimated_d_min),
                "trial": env_id,
                "success": int(clean_success),
                "raw_success": int(raw_success),
                "collision": int(collision),
                "wedge": int(wedge),
                "rejected": int(rejected),
                "oscillation_count": float(stats.oscillations[env_id].item()),
                "completion_time": float(step_count * dt),
                "min_clearance": float(stats.min_clearance[env_id].item()),
            }
        )
    wrapped.close()
    return rows


def main():
    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    all_rows = []
    for width in args_cli.widths:
        print(f"[INFO] Evaluating width={width:.3f} controller={args_cli.controller}")
        all_rows.extend(run_width(width))

    with open(args_cli.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = os.path.splitext(args_cli.output)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summarize(all_rows), f, indent=2)
    print(f"[INFO] Wrote metrics to {args_cli.output}")
    print(f"[INFO] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
