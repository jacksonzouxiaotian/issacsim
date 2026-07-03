# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate low-level narrow-passage quadruped locomotion policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate low-level narrow-passage locomotion control policies.")
parser.add_argument("--task", type=str, default="Isaac-Narrow-Gait-OracleGeometry-Anymal-C-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL checkpoint for the low-level gait policy.")
parser.add_argument("--widths", type=float, nargs="+", default=[0.75, 0.85, 0.95])
parser.add_argument(
    "--scenarios",
    type=str,
    nargs="+",
    default=["nominal"],
    choices=[
        "nominal",
        "left_wall_start",
        "right_wall_start",
        "yaw_left_start",
        "yaw_right_start",
        "doorway",
        "asymmetric_obstacle",
        "L_corridor",
    ],
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--estimated_d_min", type=float, default=0.72)
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


GENERALIZATION_TASKS = {
    "doorway": "Isaac-Narrow-Gait-Generalization-Doorway-Anymal-C-v0",
    "asymmetric_obstacle": "Isaac-Narrow-Gait-Generalization-AsymmetricObstacle-Anymal-C-v0",
    "L_corridor": "Isaac-Narrow-Gait-Generalization-LCorridor-Anymal-C-v0",
}


@dataclass
class EvalStats:
    min_clearance: torch.Tensor
    yaw_error_sum: torch.Tensor
    action_rate_sum: torch.Tensor
    action_rate_count: torch.Tensor
    oscillations: torch.Tensor
    last_lat_sign: torch.Tensor
    last_actions: torch.Tensor
    completion_step: torch.Tensor
    done: torch.Tensor
    raw_success: torch.Tensor
    collision: torch.Tensor
    wedge: torch.Tensor
    fall: torch.Tensor


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


def configure_width(env_cfg, width):
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


def configure_start_scenario(env_cfg, width, scenario):
    if scenario in ("nominal", "doorway", "asymmetric_obstacle", "L_corridor"):
        return
    velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    pose_range = {}
    if scenario == "left_wall_start":
        pose_range = {"x": (1.2, 1.8), "y": (width * 0.5 - 0.22, width * 0.5 - 0.16), "yaw": (-0.25, -0.12)}
    elif scenario == "right_wall_start":
        pose_range = {"x": (1.2, 1.8), "y": (-(width * 0.5 - 0.22), -(width * 0.5 - 0.16)), "yaw": (0.12, 0.25)}
    elif scenario == "yaw_left_start":
        pose_range = {"x": (0.6, 1.2), "y": (-0.04, 0.04), "yaw": (0.35, 0.55)}
    elif scenario == "yaw_right_start":
        pose_range = {"x": (0.6, 1.2), "y": (-0.04, 0.04), "yaw": (-0.55, -0.35)}

    if "pose_range" in env_cfg.events.reset_base.params:
        env_cfg.events.reset_base.params["pose_range"] = pose_range
        env_cfg.events.reset_base.params["velocity_range"] = velocity_range
    elif "cases" in env_cfg.events.reset_base.params:
        env_cfg.events.reset_base.params["cases"] = (
            {"weight": 1.0, "pose_range": pose_range, "velocity_range": velocity_range},
        )


def make_stats(num_envs, num_actions, device):
    return EvalStats(
        min_clearance=torch.full((num_envs,), 1.0e9, device=device),
        yaw_error_sum=torch.zeros(num_envs, device=device),
        action_rate_sum=torch.zeros(num_envs, device=device),
        action_rate_count=torch.zeros(num_envs, device=device),
        oscillations=torch.zeros(num_envs, device=device),
        last_lat_sign=torch.zeros(num_envs, device=device),
        last_actions=torch.zeros(num_envs, num_actions, device=device),
        completion_step=torch.full((num_envs,), -1, dtype=torch.long, device=device),
        done=torch.zeros(num_envs, dtype=torch.bool, device=device),
        raw_success=torch.zeros(num_envs, dtype=torch.bool, device=device),
        collision=torch.zeros(num_envs, dtype=torch.bool, device=device),
        wedge=torch.zeros(num_envs, dtype=torch.bool, device=device),
        fall=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def update_stats(stats, env, width, actions, step_idx):
    active_mask = ~stats.done
    if not bool(active_mask.any()):
        return

    robot = env.scene["robot"]
    pos = _local_root_pos(env)
    yaw_error = torch.abs(_yaw_from_quat_wxyz(robot.data.root_quat_w))
    clearance = width * 0.5 - torch.abs(pos[:, 1])
    stats.min_clearance[active_mask] = torch.minimum(stats.min_clearance[active_mask], clearance[active_mask])
    stats.yaw_error_sum[active_mask] += yaw_error[active_mask]

    action_delta = torch.linalg.norm(actions - stats.last_actions, dim=1)
    stats.action_rate_sum[active_mask] += action_delta[active_mask]
    stats.action_rate_count[active_mask] += 1.0
    stats.last_actions[active_mask] = actions[active_mask]

    lat_sign = torch.sign(robot.data.root_lin_vel_w[:, 1])
    changed = (lat_sign != 0.0) & (stats.last_lat_sign != 0.0) & (lat_sign != stats.last_lat_sign)
    stats.oscillations[active_mask] += changed[active_mask].float()
    update_lat = (lat_sign != 0.0) & active_mask
    stats.last_lat_sign[update_lat] = lat_sign[update_lat]

    speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    near_wall = clearance < 0.06
    stats.wedge |= active_mask & near_wall & (speed < 0.035) & (env.episode_length_buf > 12)

    active_terms = set(env.termination_manager.active_terms)
    if "goal_reached" in active_terms:
        stats.raw_success |= active_mask & env.termination_manager.get_term("goal_reached")
    if "base_contact" in active_terms:
        stats.collision |= active_mask & env.termination_manager.get_term("base_contact")
    if "bad_orientation" in active_terms:
        bad_orientation = env.termination_manager.get_term("bad_orientation")
        stats.fall |= active_mask & bad_orientation
        stats.collision |= active_mask & bad_orientation
    if "base_too_low" in active_terms:
        base_low = env.termination_manager.get_term("base_too_low")
        stats.fall |= active_mask & base_low
        stats.collision |= active_mask & base_low
    if "stuck" in active_terms:
        stats.wedge |= active_mask & env.termination_manager.get_term("stuck")

    just_done = env.reset_buf & active_mask
    stats.done |= env.reset_buf
    stats.completion_step[just_done] = step_idx


def get_policy(wrapped_env, task, checkpoint):
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=wrapped_env.unwrapped.device)


def run_eval(task, scenario, width):
    task_name = GENERALIZATION_TASKS.get(scenario, task)
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    configure_width(env_cfg, width)
    configure_start_scenario(env_cfg, width, scenario)

    env = gym.make(task_name, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    policy = get_policy(wrapped, task_name, args_cli.checkpoint)
    stats = make_stats(wrapped.num_envs, wrapped.num_actions, wrapped.device)

    obs = wrapped.get_observations()
    for step_idx in range(args_cli.max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)
        update_stats(stats, wrapped.unwrapped, width, actions, step_idx + 1)
        if bool(stats.done.all()):
            break

    rows = []
    dt = wrapped.unwrapped.step_dt
    for env_id in range(wrapped.num_envs):
        step_count = int(stats.completion_step[env_id].item())
        timeout = step_count < 0
        if timeout:
            step_count = args_cli.max_steps
        raw_success = bool(stats.raw_success[env_id].item())
        collision = bool(stats.collision[env_id].item())
        wedge = bool(stats.wedge[env_id].item())
        fall = bool(stats.fall[env_id].item())
        clean_success = raw_success and not collision and not wedge and not fall
        count = max(float(stats.action_rate_count[env_id].item()), 1.0)
        rows.append(
            {
                "task": task_name,
                "scenario": scenario,
                "width": float(width),
                "delta_d": float(width - args_cli.estimated_d_min),
                "trial": env_id,
                "clean_success": int(clean_success),
                "raw_success": int(raw_success),
                "collision": int(collision),
                "wedge": int(wedge),
                "timeout": int(timeout),
                "fall": int(fall),
                "time_to_goal": float(step_count * dt),
                "min_clearance": float(stats.min_clearance[env_id].item()),
                "yaw_error_mean": float(stats.yaw_error_sum[env_id].item() / count),
                "oscillation_count": float(stats.oscillations[env_id].item()),
                "action_smoothness": float(stats.action_rate_sum[env_id].item() / count),
            }
        )
    wrapped.close()
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["scenario"], row["width"])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (scenario, width), items in sorted(groups.items()):
        n = max(len(items), 1)
        complete_times = [item["time_to_goal"] for item in items if item["raw_success"]]
        summary_rows.append(
            {
                "scenario": scenario,
                "width": width,
                "num_trials": n,
                "clean_success_rate": sum(item["clean_success"] for item in items) / n,
                "raw_success_rate": sum(item["raw_success"] for item in items) / n,
                "collision_rate": sum(item["collision"] for item in items) / n,
                "wedge_rate": sum(item["wedge"] for item in items) / n,
                "timeout_rate": sum(item["timeout"] for item in items) / n,
                "mean_time_to_goal": sum(complete_times) / max(len(complete_times), 1),
                "min_clearance_mean": sum(item["min_clearance"] for item in items) / n,
                "min_clearance_min": min(item["min_clearance"] for item in items),
                "yaw_error_mean": sum(item["yaw_error_mean"] for item in items) / n,
                "oscillation_count": sum(item["oscillation_count"] for item in items) / n,
                "action_smoothness": sum(item["action_smoothness"] for item in items) / n,
                "fall_rate": sum(item["fall"] for item in items) / n,
            }
        )
    return summary_rows


def write_markdown(summary_rows, path):
    headers = [
        "scenario",
        "width",
        "clean_SR",
        "raw_SR",
        "collision",
        "wedge",
        "timeout",
        "time",
        "clear_mean",
        "clear_min",
        "yaw",
        "osc",
        "smooth",
        "fall",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join([" --- " for _ in headers]) + "|\n")
        for row in summary_rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(row["scenario"]),
                        f"{row['width']:.2f}",
                        f"{row['clean_success_rate']:.4f}",
                        f"{row['raw_success_rate']:.4f}",
                        f"{row['collision_rate']:.4f}",
                        f"{row['wedge_rate']:.4f}",
                        f"{row['timeout_rate']:.4f}",
                        f"{row['mean_time_to_goal']:.3f}",
                        f"{row['min_clearance_mean']:.3f}",
                        f"{row['min_clearance_min']:.3f}",
                        f"{row['yaw_error_mean']:.3f}",
                        f"{row['oscillation_count']:.2f}",
                        f"{row['action_smoothness']:.3f}",
                        f"{row['fall_rate']:.4f}",
                    ]
                )
                + " |\n"
            )


def main():
    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    all_rows = []
    for scenario in args_cli.scenarios:
        for width in args_cli.widths:
            print(f"[INFO] Evaluating scenario={scenario} width={width:.3f}")
            all_rows.extend(run_eval(args_cli.task, scenario, width))

    with open(args_cli.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = summarize(all_rows)
    summary_path = os.path.splitext(args_cli.output)[0] + "_summary.json"
    table_path = os.path.splitext(args_cli.output)[0] + "_table.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    write_markdown(summary_rows, table_path)

    print(f"[INFO] Wrote per-trial metrics to {args_cli.output}")
    print(f"[INFO] Wrote summary JSON to {summary_path}")
    print(f"[INFO] Wrote markdown table to {table_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
