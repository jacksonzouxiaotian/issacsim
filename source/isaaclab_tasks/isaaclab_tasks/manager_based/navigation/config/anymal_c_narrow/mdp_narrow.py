import torch


DEFAULT_ESTIMATED_D_MIN = 0.72


def _local_root_pos(env, asset_name="robot"):
    robot = env.scene[asset_name]
    pos_w = robot.data.root_pos_w[:, :3]
    env_origins = env.scene.env_origins
    return pos_w - env_origins


def _failure_memory(env):
    if not hasattr(env, "_narrow_memory"):
        env._narrow_memory = {
            "stuck_counter": torch.zeros(env.num_envs, device=env.device),
            "wedge_counter": torch.zeros(env.num_envs, device=env.device),
            "last_progress_vel": torch.zeros(env.num_envs, device=env.device),
            "last_lateral_sign": torch.zeros(env.num_envs, device=env.device),
            "oscillation_counter": torch.zeros(env.num_envs, device=env.device),
        }
    return env._narrow_memory


def reset_failure_memory(env):
    memory = _failure_memory(env)
    just_reset = env.episode_length_buf == 0
    for value in memory.values():
        value[just_reset] = 0.0
    return memory


def failure_termination_penalty(env, term_names=("stuck", "base_contact", "bad_orientation", "base_too_low")):
    """Penalize failure terminations without penalizing task success."""
    penalty = torch.zeros(env.num_envs, device=env.device)
    active_terms = set(env.termination_manager.active_terms)
    for term_name in term_names:
        if term_name in active_terms:
            penalty += env.termination_manager.get_term(term_name).float()
    return penalty


def corridor_state(
    env,
    corridor_width: float,
    corridor_length: float,
    estimated_d_min: float = DEFAULT_ESTIMATED_D_MIN,
    asset_name="robot",
):
    """Return privileged corridor geometry features.

    Output:
        [
            x_norm,
            y_norm,
            left_clearance_norm,
            right_clearance_norm,
            dist_to_goal_norm,
            delta_d_norm,
            feasible_margin_norm,
        ]
    """
    pos = _local_root_pos(env, asset_name=asset_name)
    x = pos[:, 0]
    y = pos[:, 1]

    half_w = corridor_width * 0.5
    eps = 1e-6

    left_clearance = half_w - y
    right_clearance = half_w + y
    dist_to_goal = corridor_length - x

    x_norm = x / max(corridor_length, eps)
    y_norm = y / max(half_w, eps)
    left_norm = left_clearance / max(half_w, eps)
    right_norm = right_clearance / max(half_w, eps)
    goal_norm = dist_to_goal / max(corridor_length, eps)
    delta_d = corridor_width - estimated_d_min
    delta_norm = torch.full_like(x, delta_d / max(estimated_d_min, eps))
    feasible_margin_norm = torch.minimum(left_clearance, right_clearance) / max(estimated_d_min, eps)

    return torch.stack([x_norm, y_norm, left_norm, right_norm, goal_norm, delta_norm, feasible_margin_norm], dim=-1)


def forward_progress_reward(env, asset_name="robot"):
    """Reward forward progress along corridor x direction."""
    robot = env.scene[asset_name]

    # 如果你版本里字段名不同，改这里
    vx = robot.data.root_lin_vel_w[:, 0]

    return torch.clamp(vx, min=0.0, max=1.0)


def centerline_error_l1(env, corridor_width: float, asset_name="robot"):
    """Penalty for deviating from corridor centerline."""
    pos = _local_root_pos(env, asset_name=asset_name)
    y = pos[:, 1]
    half_w = corridor_width * 0.5
    return torch.abs(y) / max(half_w, 1e-6)


def goal_reached(
    env,
    goal_x: float,
    tol: float = 0.15,
    corridor_width: float | None = None,
    lateral_margin: float = 0.08,
    asset_name="robot",
):
    """Terminate when robot reaches corridor exit."""
    pos = _local_root_pos(env, asset_name=asset_name)
    x = pos[:, 0]
    reached_x = x >= (goal_x - tol)
    if corridor_width is None:
        return reached_x
    y = pos[:, 1]
    lateral_ok = torch.abs(y) <= max(corridor_width * 0.5 - lateral_margin, 0.0)
    return reached_x & lateral_ok


def goal_reached_bonus(
    env,
    goal_x: float,
    tol: float = 0.15,
    corridor_width: float | None = None,
    lateral_margin: float = 0.08,
    asset_name="robot",
):
    """Sparse success bonus."""
    return goal_reached(
        env,
        goal_x=goal_x,
        tol=tol,
        corridor_width=corridor_width,
        lateral_margin=lateral_margin,
        asset_name=asset_name,
    ).float()


def stuck_penalty(env, min_forward_speed: float = 0.03, goal_x: float = 9.75, asset_name="robot"):
    """Penalty when robot is moving too slowly before reaching the goal."""
    robot = env.scene[asset_name]
    pos = _local_root_pos(env, asset_name=asset_name)
    x = pos[:, 0]

    vx = robot.data.root_lin_vel_w[:, 0]
    not_finished = x < goal_x
    stuck = (torch.abs(vx) < min_forward_speed) & not_finished
    return stuck.float()


def stuck_for_steps(env, window_s: float = 2.0, min_forward_speed: float = 0.02, goal_x: float = 9.75, asset_name="robot"):
    """Terminate if robot stays stuck for a sustained duration."""
    robot = env.scene[asset_name]
    pos = _local_root_pos(env, asset_name=asset_name)
    x = pos[:, 0]
    vx = robot.data.root_lin_vel_w[:, 0]

    if not hasattr(env, "_narrow_stuck_counter"):
        env._narrow_stuck_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    just_reset = env.episode_length_buf == 0
    env._narrow_stuck_counter[just_reset] = 0

    not_finished = x < goal_x
    stuck_now = (torch.abs(vx) < min_forward_speed) & not_finished

    env._narrow_stuck_counter[stuck_now] += 1
    env._narrow_stuck_counter[~stuck_now] = 0

    step_dt = env.cfg.sim.dt * env.cfg.decimation
    stuck_steps = max(1, int(window_s / step_dt))

    return env._narrow_stuck_counter >= stuck_steps


def unfinished_time_penalty(env, goal_x: float, asset_name="robot"):
    """Penalty per step if the robot has not yet reached the goal."""
    pos = _local_root_pos(env, asset_name=asset_name)
    x = pos[:, 0]
    not_finished = x < goal_x
    return not_finished.float()


def clearance_margin(env, corridor_width: float, estimated_d_min: float = DEFAULT_ESTIMATED_D_MIN, asset_name="robot"):
    """Return normalized remaining clearance around the base."""
    pos = _local_root_pos(env, asset_name=asset_name)
    y = pos[:, 1]
    half_w = corridor_width * 0.5
    min_wall_clearance = half_w - torch.abs(y)
    return min_wall_clearance / max(estimated_d_min, 1e-6)


def unsafe_clearance_penalty(
    env,
    corridor_width: float,
    safety_margin: float = 0.08,
    asset_name="robot",
):
    """Penalize approaching the walls before hard contact happens."""
    pos = _local_root_pos(env, asset_name=asset_name)
    y = pos[:, 1]
    half_w = corridor_width * 0.5
    clearance = half_w - torch.abs(y)
    return torch.clamp(safety_margin - clearance, min=0.0) / max(safety_margin, 1e-6)


def recovery_memory_state(
    env,
    corridor_width: float,
    estimated_d_min: float = DEFAULT_ESTIMATED_D_MIN,
    asset_name: str = "robot",
):
    """Return compact failure-aware memory for a feed-forward policy.

    Output:
        [stuck_norm, wedge_norm, oscillation_norm, min_clearance_norm, delta_d_norm]
    """
    memory = reset_failure_memory(env)
    robot = env.scene[asset_name]
    pos = _local_root_pos(env, asset_name=asset_name)
    y = pos[:, 1]
    vx = robot.data.root_lin_vel_w[:, 0]
    vy = robot.data.root_lin_vel_w[:, 1]
    speed_xy = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    half_w = corridor_width * 0.5
    clearance = half_w - torch.abs(y)
    unsafe = clearance < 0.08
    stuck = speed_xy < 0.04

    memory["stuck_counter"][stuck] += 1.0
    memory["stuck_counter"][~stuck] = 0.0
    memory["wedge_counter"][unsafe & stuck] += 1.0
    memory["wedge_counter"][~(unsafe & stuck)] = 0.0

    lateral_sign = torch.sign(vy)
    changed_sign = (lateral_sign != 0.0) & (memory["last_lateral_sign"] != 0.0) & (lateral_sign != memory["last_lateral_sign"])
    memory["oscillation_counter"][changed_sign] += 1.0
    memory["last_lateral_sign"][lateral_sign != 0.0] = lateral_sign[lateral_sign != 0.0]
    memory["last_progress_vel"][:] = vx

    delta_d = corridor_width - estimated_d_min
    delta_norm = torch.full_like(y, delta_d / max(estimated_d_min, 1e-6))

    return torch.stack(
        [
            torch.clamp(memory["stuck_counter"] / 20.0, max=1.0),
            torch.clamp(memory["wedge_counter"] / 20.0, max=1.0),
            torch.clamp(memory["oscillation_counter"] / 20.0, max=1.0),
            clearance / max(estimated_d_min, 1e-6),
            delta_norm,
        ],
        dim=-1,
    )


def recovery_progress_reward(
    env,
    min_stuck_steps: float = 5.0,
    asset_name: str = "robot",
):
    """Reward positive progress after the policy has recently been stuck."""
    memory = reset_failure_memory(env)
    robot = env.scene[asset_name]
    vx = robot.data.root_lin_vel_w[:, 0]
    was_stuck = memory["stuck_counter"] >= min_stuck_steps
    return torch.where(was_stuck, torch.clamp(vx, min=0.0, max=0.5), torch.zeros_like(vx))


def oscillation_penalty(env):
    """Small penalty for repeated lateral sign changes."""
    memory = reset_failure_memory(env)
    return torch.clamp(memory["oscillation_counter"] / 20.0, max=1.0)


# =============================================================================
# L-shaped corridor helpers
# =============================================================================


def _l_corridor_in_second_segment(x: torch.Tensor, y: torch.Tensor, corner_x: float, corridor_width: float | None = None):
    if corridor_width is None:
        return (x >= corner_x) & (y >= -0.10)
    corner_gate = max(0.10, corridor_width * 0.25)
    return (x >= corner_x - corner_gate) & (y >= -corner_gate)


def _l_corridor_progress_from_xy(x: torch.Tensor, y: torch.Tensor, corner_x: float, goal_y: float):
    """Compute approximate path progress along an L-shaped centerline.

    Segment 1:
        (0, 0) -> (corner_x, 0), progress = x

    Segment 2:
        (corner_x, 0) -> (corner_x, goal_y), progress = corner_x + y
    """
    in_second_segment = _l_corridor_in_second_segment(x, y, corner_x)

    progress_first = torch.clamp(x, min=0.0, max=corner_x)
    progress_second = corner_x + torch.clamp(y, min=0.0, max=goal_y)

    return torch.where(in_second_segment, progress_second, progress_first)


def l_corridor_state(
    env,
    corridor_width: float,
    corner_x: float,
    goal_y: float,
    estimated_d_min: float = DEFAULT_ESTIMATED_D_MIN,
    asset_name: str = "robot",
):
    """Return privileged L-corridor geometry features.

    Output:
        [
            progress_norm,
            centerline_error_norm,
            dist_to_corner_norm,
            dist_to_goal_norm,
            local_heading_target_cos,
            local_heading_target_sin,
            delta_d_norm,
            min_clearance_norm,
        ]

    Explanation:
        - Before the corner, centerline is y = 0 and desired heading is +x.
        - After the corner, centerline is x = corner_x and desired heading is +y.
    """
    pos = _local_root_pos(env, asset_name=asset_name)
    xy = pos[:, :2]
    x = xy[:, 0]
    y = xy[:, 1]

    eps = 1e-6
    total_path_len = max(corner_x + goal_y, eps)
    half_w = max(corridor_width * 0.5, eps)

    in_second_segment = _l_corridor_in_second_segment(x, y, corner_x, corridor_width)

    # Path progress.
    progress = _l_corridor_progress_from_xy(
        x=x,
        y=y,
        corner_x=corner_x,
        goal_y=goal_y,
    )
    progress_norm = progress / total_path_len

    # Centerline error.
    error_first = torch.abs(y)
    error_second = torch.abs(x - corner_x)
    center_error = torch.where(in_second_segment, error_second, error_first)
    center_error_norm = center_error / half_w

    # Distance to corner and goal.
    corner = xy.new_tensor([corner_x, 0.0]).unsqueeze(0)
    goal = xy.new_tensor([corner_x, goal_y]).unsqueeze(0)

    dist_corner = torch.norm(xy - corner, dim=1)
    dist_goal = torch.norm(xy - goal, dim=1)

    dist_corner_norm = dist_corner / total_path_len
    dist_goal_norm = dist_goal / total_path_len

    # Local target heading.
    # Segment 1: heading 0 rad, cos=1, sin=0.
    # Segment 2: heading pi/2 rad, cos=0, sin=1.
    heading_cos_first = torch.ones_like(x)
    heading_sin_first = torch.zeros_like(x)

    heading_cos_second = torch.zeros_like(x)
    heading_sin_second = torch.ones_like(x)

    heading_cos = torch.where(in_second_segment, heading_cos_second, heading_cos_first)
    heading_sin = torch.where(in_second_segment, heading_sin_second, heading_sin_first)
    delta_d = corridor_width - estimated_d_min
    delta_norm = torch.full_like(x, delta_d / max(estimated_d_min, eps))
    min_clearance = torch.clamp(half_w - center_error, max=half_w)
    min_clearance_norm = min_clearance / max(estimated_d_min, eps)

    return torch.stack(
        [
            progress_norm,
            center_error_norm,
            dist_corner_norm,
            dist_goal_norm,
            heading_cos,
            heading_sin,
            delta_norm,
            min_clearance_norm,
        ],
        dim=-1,
    )


def l_corridor_path_progress(
    env,
    corner_x: float,
    goal_y: float,
    corridor_width: float | None = None,
    asset_name: str = "robot",
):
    """Reward forward progress along the L-shaped corridor path.

    This version uses velocity projected onto the local desired direction:
        - before corner: reward positive vx
        - after corner: reward positive vy

    This is usually more stable than returning absolute progress.
    """
    robot = env.scene[asset_name]
    pos = _local_root_pos(env, asset_name=asset_name)

    x = pos[:, 0]
    vx = robot.data.root_lin_vel_w[:, 0]
    vy = robot.data.root_lin_vel_w[:, 1]

    y = pos[:, 1]
    in_second_segment = _l_corridor_in_second_segment(x, y, corner_x, corridor_width)

    progress_vel = torch.where(in_second_segment, vy, vx)

    return torch.clamp(progress_vel, min=0.0, max=1.0)


def l_corridor_progress_dense(
    env,
    corner_x: float,
    goal_y: float,
    corridor_width: float | None = None,
    asset_name: str = "robot",
):
    """Dense reward for absolute path progress along the L-corridor."""
    pos = _local_root_pos(env, asset_name=asset_name)
    x = pos[:, 0]
    y = pos[:, 1]
    progress = _l_corridor_progress_from_xy(x=x, y=y, corner_x=corner_x, goal_y=goal_y)
    return torch.clamp(progress / max(corner_x + goal_y, 1e-6), min=0.0, max=1.0)


def l_corridor_corner_reached_bonus(
    env,
    corner_x: float,
    tol: float = 0.35,
    asset_name: str = "robot",
):
    """Intermediate sparse bonus for reaching the turn entrance."""
    pos = _local_root_pos(env, asset_name=asset_name)
    xy = pos[:, :2]
    corner = xy.new_tensor([corner_x, 0.0]).unsqueeze(0)
    dist = torch.norm(xy - corner, dim=1)
    return (dist <= tol).float()


def l_corridor_goal_distance_tanh(
    env,
    goal_x: float,
    goal_y: float,
    std: float = 1.5,
    asset_name: str = "robot",
):
    """Dense reward for getting closer to the final local passage goal."""
    pos = _local_root_pos(env, asset_name=asset_name)
    xy = pos[:, :2]
    goal = xy.new_tensor([goal_x, goal_y]).unsqueeze(0)
    dist = torch.norm(xy - goal, dim=1)
    return 1.0 - torch.tanh(dist / max(std, 1e-6))


def l_corridor_centerline_error(
    env,
    corner_x: float,
    corridor_width: float | None = None,
    asset_name: str = "robot",
):
    """Penalty for deviating from the L-shaped corridor centerline.

    Segment 1:
        centerline y = 0

    Segment 2:
        centerline x = corner_x
    """
    pos = _local_root_pos(env, asset_name=asset_name)
    x = pos[:, 0]
    y = pos[:, 1]

    in_second_segment = _l_corridor_in_second_segment(x, y, corner_x, corridor_width)

    error_first = torch.abs(y)
    error_second = torch.abs(x - corner_x)

    return torch.where(in_second_segment, error_second, error_first)


def goal_reached_xy(
    env,
    goal_x: float,
    goal_y: float,
    tol: float = 0.20,
    asset_name: str = "robot",
):
    """Terminate when robot reaches a 2D goal."""
    pos = _local_root_pos(env, asset_name=asset_name)
    xy = pos[:, :2]

    goal = xy.new_tensor([goal_x, goal_y]).unsqueeze(0)
    dist = torch.norm(xy - goal, dim=1)

    return dist <= tol


def goal_reached_bonus_xy(
    env,
    goal_x: float,
    goal_y: float,
    tol: float = 0.20,
    asset_name: str = "robot",
):
    """Sparse success bonus for reaching a 2D goal."""
    return goal_reached_xy(
        env,
        goal_x=goal_x,
        goal_y=goal_y,
        tol=tol,
        asset_name=asset_name,
    ).float()


def unfinished_time_penalty_xy(
    env,
    goal_x: float,
    goal_y: float,
    tol: float = 0.20,
    asset_name: str = "robot",
):
    """Penalty per step if the robot has not yet reached the 2D goal."""
    reached = goal_reached_xy(
        env,
        goal_x=goal_x,
        goal_y=goal_y,
        tol=tol,
        asset_name=asset_name,
    )
    return (~reached).float()    
