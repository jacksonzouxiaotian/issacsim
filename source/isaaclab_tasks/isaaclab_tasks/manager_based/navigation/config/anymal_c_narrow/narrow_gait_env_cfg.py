# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level narrow-passage locomotion control task for ANYmal-C.

This task trains the quadruped controller directly with PPO. The action is a
12D joint position target. The policy observes proprioception and compact local
geometry; it does not use memory or a high-level navigation policy.
"""

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from . import mdp_narrow as narrow_mdp

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import AnymalCFlatEnvCfg


CORRIDOR_WIDTH = 0.85
CORRIDOR_LENGTH = 5.0
CORRIDOR_WALL_HEIGHT = 1.2
CORRIDOR_WALL_THICKNESS = 0.08
ESTIMATED_D_MIN = 0.72

GOAL_X = CORRIDOR_LENGTH - 0.20
GOAL_TOL = 0.20

RECOVERY_RESET_MILD_CASES = (
    {
        "weight": 0.70,
        "pose_range": {"x": (-0.9, -0.5), "y": (-0.05, 0.05), "yaw": (-0.10, 0.10)},
        "velocity_range": {
            "x": (0.0, 0.05),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    },
    {
        "weight": 0.10,
        "pose_range": {"x": (0.3, 1.6), "y": (0.07, 0.13), "yaw": (-0.12, -0.03)},
        "velocity_range": {
            "x": (0.00, 0.08),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.03, 0.03),
        },
    },
    {
        "weight": 0.10,
        "pose_range": {"x": (0.3, 1.6), "y": (-0.13, -0.07), "yaw": (0.03, 0.12)},
        "velocity_range": {
            "x": (0.00, 0.08),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.03, 0.03),
        },
    },
    {
        "weight": 0.10,
        "pose_range": {"x": (0.3, 1.5), "y": (-0.04, 0.04), "yaw": (-0.16, 0.16)},
        "velocity_range": {
            "x": (0.00, 0.08),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.03, 0.03),
        },
    },
)

RECOVERY_RESET_RIGHT_WALL_SMALL_YAW_CASES = (
    {
        "weight": 0.15,
        "pose_range": {"x": (-0.9, -0.5), "y": (-0.04, 0.04), "yaw": (-0.08, 0.08)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[0]["velocity_range"],
    },
    {
        "weight": 0.50,
        "pose_range": {"x": (0.35, 2.0), "y": (-0.24, -0.16), "yaw": (0.05, 0.18)},
        "velocity_range": {
            "x": (0.00, 0.06),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.025, 0.025),
        },
    },
    {
        "weight": 0.15,
        "pose_range": {"x": (0.35, 1.8), "y": (-0.04, 0.04), "yaw": (-0.20, 0.20)},
        "velocity_range": {
            "x": (0.00, 0.07),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.03, 0.03),
        },
    },
    {
        "weight": 0.10,
        "pose_range": {"x": (0.35, 1.8), "y": (0.10, 0.18), "yaw": (-0.14, -0.04)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[1]["velocity_range"],
    },
    {
        "weight": 0.10,
        "pose_range": {"x": (0.35, 1.8), "y": (-0.18, -0.10), "yaw": (0.04, 0.14)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[2]["velocity_range"],
    },
)

RECOVERY_RESET_MEDIUM_CASES = (
    {
        "weight": 0.40,
        "pose_range": {"x": (-0.9, -0.5), "y": (-0.05, 0.05), "yaw": (-0.10, 0.10)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[0]["velocity_range"],
    },
    {
        "weight": 0.22,
        "pose_range": {"x": (0.5, 2.6), "y": (0.14, 0.24), "yaw": (-0.24, -0.08)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[1]["velocity_range"],
    },
    {
        "weight": 0.22,
        "pose_range": {"x": (0.5, 2.6), "y": (-0.24, -0.14), "yaw": (0.08, 0.24)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[2]["velocity_range"],
    },
    {
        "weight": 0.16,
        "pose_range": {"x": (0.4, 2.3), "y": (-0.06, 0.06), "yaw": (-0.38, 0.38)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[3]["velocity_range"],
    },
)

RECOVERY_RESET_HARD_CASES = (
    {
        "weight": 0.35,
        "pose_range": {"x": (-0.9, -0.5), "y": (-0.05, 0.05), "yaw": (-0.10, 0.10)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[0]["velocity_range"],
    },
    {
        "weight": 0.25,
        "pose_range": {"x": (0.6, 2.8), "y": (0.18, 0.29), "yaw": (-0.32, -0.10)},
        "velocity_range": {
            "x": (-0.03, 0.06),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.08, 0.08),
        },
    },
    {
        "weight": 0.25,
        "pose_range": {"x": (0.6, 2.8), "y": (-0.29, -0.18), "yaw": (0.10, 0.32)},
        "velocity_range": {
            "x": (-0.03, 0.06),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.08, 0.08),
        },
    },
    {
        "weight": 0.15,
        "pose_range": {"x": (0.4, 2.4), "y": (-0.08, 0.08), "yaw": (-0.55, 0.55)},
        "velocity_range": {
            "x": (-0.04, 0.04),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.12, 0.12),
        },
    },
)

RECOVERY_RESET_BALANCED_CLEAN_CASES = (
    {
        "weight": 0.20,
        "pose_range": {"x": (-0.9, -0.5), "y": (-0.05, 0.05), "yaw": (-0.10, 0.10)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[0]["velocity_range"],
    },
    {
        "weight": 0.18,
        "pose_range": {"x": (0.6, 2.7), "y": (0.16, 0.27), "yaw": (-0.28, -0.08)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[1]["velocity_range"],
    },
    {
        "weight": 0.18,
        "pose_range": {"x": (0.6, 2.7), "y": (-0.27, -0.16), "yaw": (0.08, 0.28)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[2]["velocity_range"],
    },
    {
        "weight": 0.22,
        "pose_range": {"x": (0.4, 2.4), "y": (-0.05, 0.05), "yaw": (0.34, 0.62)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[3]["velocity_range"],
    },
    {
        "weight": 0.22,
        "pose_range": {"x": (0.4, 2.4), "y": (-0.05, 0.05), "yaw": (-0.62, -0.34)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[3]["velocity_range"],
    },
)

RECOVERY_RESET_YAW_CLEAN_CASES = (
    {
        "weight": 0.15,
        "pose_range": {"x": (-0.9, -0.5), "y": (-0.04, 0.04), "yaw": (-0.08, 0.08)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[0]["velocity_range"],
    },
    {
        "weight": 0.12,
        "pose_range": {"x": (0.6, 2.4), "y": (0.14, 0.24), "yaw": (-0.24, -0.08)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[1]["velocity_range"],
    },
    {
        "weight": 0.12,
        "pose_range": {"x": (0.6, 2.4), "y": (-0.24, -0.14), "yaw": (0.08, 0.24)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[2]["velocity_range"],
    },
    {
        "weight": 0.305,
        "pose_range": {"x": (0.4, 2.2), "y": (-0.05, 0.05), "yaw": (0.34, 0.64)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[3]["velocity_range"],
    },
    {
        "weight": 0.305,
        "pose_range": {"x": (0.4, 2.2), "y": (-0.05, 0.05), "yaw": (-0.64, -0.34)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[3]["velocity_range"],
    },
)

RECOVERY_RESET_NEAR_WALL_CLEAN_CASES = (
    {
        "weight": 0.15,
        "pose_range": {"x": (-0.9, -0.5), "y": (-0.04, 0.04), "yaw": (-0.08, 0.08)},
        "velocity_range": RECOVERY_RESET_MILD_CASES[0]["velocity_range"],
    },
    {
        "weight": 0.425,
        "pose_range": {"x": (0.7, 2.8), "y": (0.15, 0.27), "yaw": (-0.26, -0.06)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[1]["velocity_range"],
    },
    {
        "weight": 0.425,
        "pose_range": {"x": (0.7, 2.8), "y": (-0.27, -0.15), "yaw": (0.06, 0.26)},
        "velocity_range": RECOVERY_RESET_HARD_CASES[2]["velocity_range"],
    },
)

RECOVERY_RESET_CASES = RECOVERY_RESET_HARD_CASES


@configclass
class NarrowGaitEnvCfg(AnymalCFlatEnvCfg):
    """Direct joint-action locomotion policy for local narrow passage traversal."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4096
        self.scene.env_spacing = 8.0
        self.episode_length_s = 12.0

        # Use the flat terrain importer as the floor, then add per-env corridor walls.
        self.scene.left_wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/LeftWall",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(
                    CORRIDOR_LENGTH / 2.0,
                    -(CORRIDOR_WIDTH / 2.0 + CORRIDOR_WALL_THICKNESS / 2.0),
                    CORRIDOR_WALL_HEIGHT / 2.0,
                )
            ),
            spawn=sim_utils.CuboidCfg(
                size=(CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS, CORRIDOR_WALL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.70, 0.72)),
            ),
        )
        self.scene.right_wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RightWall",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(
                    CORRIDOR_LENGTH / 2.0,
                    +(CORRIDOR_WIDTH / 2.0 + CORRIDOR_WALL_THICKNESS / 2.0),
                    CORRIDOR_WALL_HEIGHT / 2.0,
                )
            ),
            spawn=sim_utils.CuboidCfg(
                size=(CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS, CORRIDOR_WALL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.70, 0.72)),
            ),
        )

        # Local-passage command: move forward with near-zero lateral/yaw command.
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.03, 0.03)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.05, 0.05)
        self.commands.base_velocity.ranges.heading = (-0.08, 0.08)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.resampling_time_range = (12.0, 12.0)

        # Start before the entrance with small pose noise.
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.9, -0.5), "y": (-0.04, 0.04), "yaw": (-0.08, 0.08)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
        self.events.push_robot = None

        # Add compact local geometry to the gait policy observation.
        self.observations.policy.corridor_state = ObsTerm(
            func=narrow_mdp.corridor_state,
            params={
                "corridor_width": CORRIDOR_WIDTH,
                "corridor_length": CORRIDOR_LENGTH,
                "estimated_d_min": ESTIMATED_D_MIN,
            },
        )
        self.observations.policy.enable_corruption = True
        self.observations.policy.base_lin_vel.noise = Unoise(n_min=-0.08, n_max=0.08)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.15, n_max=0.15)

        # Retune locomotion rewards for narrow-passage gait rather than generic velocity tracking.
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35
        self.rewards.track_ang_vel_z_exp.weight = 0.15
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.feet_air_time.weight = 0.35
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.undesired_contacts.weight = -5.0

        self.rewards.forward_progress = RewTerm(func=narrow_mdp.forward_progress_reward, weight=4.0)
        self.rewards.centerline_penalty = RewTerm(
            func=narrow_mdp.centerline_error_l1,
            weight=-3.0,
            params={"corridor_width": CORRIDOR_WIDTH},
        )
        self.rewards.unsafe_clearance = RewTerm(
            func=narrow_mdp.unsafe_clearance_penalty,
            weight=-8.0,
            params={"corridor_width": CORRIDOR_WIDTH, "safety_margin": 0.14},
        )
        self.rewards.yaw_alignment = RewTerm(func=narrow_mdp.heading_error_abs, weight=-4.0)
        self.rewards.lateral_velocity = RewTerm(func=narrow_mdp.lateral_velocity_penalty, weight=-0.8)
        self.rewards.base_height = RewTerm(
            func=narrow_mdp.base_height_error_l1,
            weight=-1.5,
            params={"target_height": 0.50},
        )
        self.rewards.time_penalty = RewTerm(
            func=narrow_mdp.unfinished_time_penalty,
            weight=-0.04,
            params={"goal_x": GOAL_X},
        )
        self.rewards.success_bonus = RewTerm(
            func=narrow_mdp.clean_goal_reached_bonus,
            weight=220.0,
            params={
                "goal_x": GOAL_X,
                "tol": GOAL_TOL,
                "corridor_width": CORRIDOR_WIDTH,
                "lateral_margin": 0.08,
            },
        )
        self.rewards.failure_termination = RewTerm(func=narrow_mdp.failure_termination_penalty, weight=-250.0)

        self.terminations.goal_reached = DoneTerm(
            func=narrow_mdp.goal_reached,
            params={
                "goal_x": GOAL_X,
                "tol": GOAL_TOL,
                "corridor_width": CORRIDOR_WIDTH,
                "lateral_margin": 0.08,
            },
        )
        self.terminations.stuck = DoneTerm(
            func=narrow_mdp.stuck_for_steps,
            params={"window_s": 1.0, "min_forward_speed": 0.03, "goal_x": GOAL_X},
        )
        self.terminations.base_contact.params["threshold"] = 8.0


@configclass
class NarrowGaitEnvCfg_PLAY(NarrowGaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False


@configclass
class NarrowGaitOracleGeometryEnvCfg(NarrowGaitEnvCfg):
    """Straight corridor with privileged analytic geometry observation."""


@configclass
class NarrowGaitSensorEstimatedGeometryEnvCfg(NarrowGaitEnvCfg):
    """Straight corridor with noisy sensor-estimated geometry observation."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.corridor_state = ObsTerm(
            func=narrow_mdp.sensor_estimated_corridor_state,
            params={
                "corridor_width": CORRIDOR_WIDTH,
                "corridor_length": CORRIDOR_LENGTH,
                "estimated_d_min": ESTIMATED_D_MIN,
            },
        )


@configclass
class NarrowGaitGeneralizationDoorwayEnvCfg(NarrowGaitSensorEstimatedGeometryEnvCfg):
    """Doorway generalization scene with a local constriction."""

    def __post_init__(self):
        super().__post_init__()
        jamb_thickness = 0.12
        jamb_depth = 0.28
        doorway_width = 0.76
        jamb_y = doorway_width / 2.0 + jamb_thickness / 2.0
        self.scene.doorway_left_jamb = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DoorwayLeftJamb",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.35, -jamb_y, CORRIDOR_WALL_HEIGHT / 2.0)),
            spawn=sim_utils.CuboidCfg(
                size=(jamb_depth, jamb_thickness, CORRIDOR_WALL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.58, 0.66)),
            ),
        )
        self.scene.doorway_right_jamb = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/DoorwayRightJamb",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.35, jamb_y, CORRIDOR_WALL_HEIGHT / 2.0)),
            spawn=sim_utils.CuboidCfg(
                size=(jamb_depth, jamb_thickness, CORRIDOR_WALL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.58, 0.66)),
            ),
        )


@configclass
class NarrowGaitGeneralizationAsymmetricObstacleEnvCfg(NarrowGaitSensorEstimatedGeometryEnvCfg):
    """Asymmetric obstacle generalization scene with one-sided protrusions."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.right_protrusion = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/RightProtrusion",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(2.00, CORRIDOR_WIDTH / 2.0 - 0.05, CORRIDOR_WALL_HEIGHT / 2.0)
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.55, 0.18, CORRIDOR_WALL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.52, 0.44)),
            ),
        )
        self.scene.left_protrusion = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/LeftProtrusion",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(3.25, -(CORRIDOR_WIDTH / 2.0 - 0.08), CORRIDOR_WALL_HEIGHT / 2.0)
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.42, 0.14, CORRIDOR_WALL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.52, 0.44)),
            ),
        )


@configclass
class NarrowGaitGeneralizationLCorridorEnvCfg(NarrowGaitSensorEstimatedGeometryEnvCfg):
    """L-turn scene entry for generalization experiments.

    The low-level policy observes compact local geometry; this scene is an
    evaluation scaffold for locomotion-control generalization.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.corner_block = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/LCornerBlock",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(3.15, 0.18, CORRIDOR_WALL_HEIGHT / 2.0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.40, 0.62, CORRIDOR_WALL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.54, 0.50)),
            ),
        )
        self.scene.turn_guide_wall = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/LTurnGuideWall",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(3.55, -0.46, CORRIDOR_WALL_HEIGHT / 2.0)),
            spawn=sim_utils.CuboidCfg(
                size=(1.10, CORRIDOR_WALL_THICKNESS, CORRIDOR_WALL_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.54, 0.50)),
            ),
        )


@configclass
class NarrowGaitRecoveryEnvCfg(NarrowGaitEnvCfg):
    """Fine-tuning task that mixes entrance traversal with near-wall/yawed starts."""

    recovery_reset_cases = RECOVERY_RESET_HARD_CASES

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_base = EventTerm(
            func=narrow_mdp.reset_root_state_corridor_recovery,
            mode="reset",
            params={"cases": self.recovery_reset_cases},
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.55)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.02, 0.02)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.03, 0.03)
        self.commands.base_velocity.ranges.heading = (-0.05, 0.05)

        self.rewards.track_lin_vel_xy_exp.weight = 0.95
        self.rewards.forward_progress.weight = 3.2
        self.rewards.centerline_penalty.weight = -2.8
        self.rewards.unsafe_clearance.weight = -14.0
        self.rewards.unsafe_clearance.params["safety_margin"] = 0.22
        self.rewards.success_bonus.weight = 260.0
        self.rewards.time_penalty.weight = -0.08
        self.rewards.failure_termination.weight = -420.0
        self.rewards.heading_alignment = RewTerm(func=narrow_mdp.heading_error_abs, weight=-5.0)
        self.rewards.current_realign = RewTerm(
            func=narrow_mdp.recovery_realign_reward,
            weight=1.8,
            params={"corridor_width": CORRIDOR_WIDTH, "goal_x": GOAL_X},
        )
        self.rewards.clearance_recovery = RewTerm(
            func=narrow_mdp.wall_escape_reward,
            weight=2.0,
            params={"corridor_width": CORRIDOR_WIDTH, "near_wall_threshold": 0.24},
        )
        self.rewards.centerline_velocity = RewTerm(
            func=narrow_mdp.centerline_velocity_reward,
            weight=2.5,
            params={"corridor_width": CORRIDOR_WIDTH, "near_wall_threshold": 0.24},
        )
        self.rewards.yaw_correction = RewTerm(func=narrow_mdp.yaw_correction_reward, weight=1.2)
        self.rewards.lateral_velocity.weight = -1.0

        self.terminations.stuck.params["window_s"] = 1.2
        self.terminations.stuck.params["min_forward_speed"] = 0.035


@configclass
class NarrowGaitRecoveryMildEnvCfg(NarrowGaitRecoveryEnvCfg):
    recovery_reset_cases = RECOVERY_RESET_MILD_CASES


@configclass
class NarrowGaitRecoveryRightWallSmallYawEnvCfg(NarrowGaitRecoveryEnvCfg):
    """Right-wall and small-yaw recovery phase with contact-safe passage pressure."""

    recovery_reset_cases = RECOVERY_RESET_RIGHT_WALL_SMALL_YAW_CASES

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.42)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.01, 0.01)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.015, 0.015)
        self.commands.base_velocity.ranges.heading = (-0.04, 0.04)

        self.rewards.track_lin_vel_xy_exp.weight = 0.85
        self.rewards.forward_progress.weight = 2.8
        self.rewards.centerline_penalty.weight = -2.4
        self.rewards.unsafe_clearance.weight = -24.0
        self.rewards.unsafe_clearance.params["safety_margin"] = 0.24
        self.rewards.success_bonus.weight = 300.0
        self.rewards.failure_termination.weight = -760.0
        self.rewards.heading_alignment.weight = -5.5
        self.rewards.current_realign.weight = 1.2
        self.rewards.clearance_recovery.weight = 4.0
        self.rewards.centerline_velocity.weight = 3.5
        self.rewards.yaw_correction.weight = 1.4
        self.rewards.lateral_velocity.weight = -1.4
        self.terminations.stuck.params["window_s"] = 1.0
        self.terminations.stuck.params["min_forward_speed"] = 0.04


@configclass
class NarrowGaitAblationFullRewardEnvCfg(NarrowGaitRecoveryRightWallSmallYawEnvCfg):
    """Ablation reference: full reward with recovery-start curriculum."""


@configclass
class NarrowGaitAblationNoClearanceRewardEnvCfg(NarrowGaitRecoveryRightWallSmallYawEnvCfg):
    """Ablation: remove the unsafe-clearance reward term only."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.unsafe_clearance.weight = 0.0


@configclass
class NarrowGaitAblationNoCenterlineRewardEnvCfg(NarrowGaitRecoveryRightWallSmallYawEnvCfg):
    """Ablation: remove the centerline-tracking reward term only."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.centerline_penalty.weight = 0.0


@configclass
class NarrowGaitAblationNoRecoveryCurriculumEnvCfg(NarrowGaitEnvCfg):
    """Ablation: full low-level reward with entrance-only resets."""


@configclass
class NarrowGaitRecoveryMediumEnvCfg(NarrowGaitRecoveryEnvCfg):
    recovery_reset_cases = RECOVERY_RESET_MEDIUM_CASES


@configclass
class NarrowGaitRecoveryHardEnvCfg(NarrowGaitRecoveryEnvCfg):
    recovery_reset_cases = RECOVERY_RESET_HARD_CASES


@configclass
class NarrowGaitRecoveryHardCleanRewardEnvCfg(NarrowGaitRecoveryHardEnvCfg):
    """Hard recovery fine-tune with stricter clean-success pressure."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 0.85
        self.rewards.forward_progress.weight = 2.4
        self.rewards.undesired_contacts.weight = -6.0
        self.rewards.unsafe_clearance.weight = -16.0
        self.rewards.success_bonus.weight = 300.0
        self.rewards.failure_termination.weight = -700.0
        self.rewards.current_realign.weight = 1.2
        self.rewards.clearance_recovery.weight = 2.4
        self.rewards.centerline_velocity.weight = 2.2
        self.rewards.yaw_correction.weight = 1.0
        self.rewards.lateral_velocity.weight = -1.3


@configclass
class NarrowGaitRecoveryBalancedCleanEnvCfg(NarrowGaitRecoveryHardCleanRewardEnvCfg):
    """Balanced clean recovery curriculum with explicit yaw-realignment starts."""

    recovery_reset_cases = RECOVERY_RESET_BALANCED_CLEAN_CASES

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.18, 0.45)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.02, 0.02)
        self.rewards.forward_progress = RewTerm(
            func=narrow_mdp.contact_free_forward_progress_reward,
            weight=1.0,
            params={"corridor_width": CORRIDOR_WIDTH, "min_clearance": 0.16, "max_abs_yaw": 0.35},
        )
        self.rewards.heading_alignment.weight = -6.0
        self.rewards.current_realign.weight = 1.4
        self.rewards.clearance_recovery.weight = 3.0
        self.rewards.centerline_velocity.weight = 3.0
        self.rewards.yaw_correction.weight = 1.4
        self.rewards.lateral_velocity.weight = -1.5


@configclass
class NarrowGaitRecoveryYawCleanEnvCfg(NarrowGaitRecoveryHardCleanRewardEnvCfg):
    """Yaw-heavy recovery phase that trains reorientation before forward passage."""

    recovery_reset_cases = RECOVERY_RESET_YAW_CLEAN_CASES

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.05, 0.30)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.01, 0.01)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.01, 0.01)
        self.rewards.track_lin_vel_xy_exp.weight = 0.75
        self.rewards.forward_progress = RewTerm(
            func=narrow_mdp.contact_free_forward_progress_reward,
            weight=1.8,
            params={"corridor_width": CORRIDOR_WIDTH, "min_clearance": 0.13, "max_abs_yaw": 0.45},
        )
        self.rewards.centerline_penalty.weight = -2.4
        self.rewards.heading_alignment.weight = -7.0
        self.rewards.current_realign.weight = 1.2
        self.rewards.yaw_realign = RewTerm(
            func=narrow_mdp.yaw_realign_progress_reward,
            weight=2.5,
            params={"goal_x": GOAL_X},
        )
        self.rewards.yaw_correction.weight = 2.0
        self.rewards.centerline_velocity.weight = 2.0
        self.rewards.clearance_recovery.weight = 2.5
        self.rewards.lateral_velocity.weight = -1.4


@configclass
class NarrowGaitRecoveryNearWallCleanEnvCfg(NarrowGaitRecoveryHardCleanRewardEnvCfg):
    """Near-wall recovery phase focused on contact-free wall escape."""

    recovery_reset_cases = RECOVERY_RESET_NEAR_WALL_CLEAN_CASES

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.12, 0.38)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.01, 0.01)
        self.rewards.forward_progress = RewTerm(
            func=narrow_mdp.contact_free_forward_progress_reward,
            weight=1.6,
            params={"corridor_width": CORRIDOR_WIDTH, "min_clearance": 0.13, "max_abs_yaw": 0.42},
        )
        self.rewards.centerline_penalty.weight = -2.6
        self.rewards.unsafe_clearance.weight = -22.0
        self.rewards.current_realign.weight = 1.2
        self.rewards.clearance_recovery.weight = 3.5
        self.rewards.centerline_velocity.weight = 3.0
        self.rewards.heading_alignment.weight = -5.0
        self.rewards.yaw_correction.weight = 1.3
        self.rewards.lateral_velocity.weight = -1.5


@configclass
class NarrowGaitRecoveryHardSensorEstimatedGeometryEnvCfg(NarrowGaitRecoveryEnvCfg):
    recovery_reset_cases = RECOVERY_RESET_HARD_CASES

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.corridor_state = ObsTerm(
            func=narrow_mdp.sensor_estimated_corridor_state,
            params={
                "corridor_width": CORRIDOR_WIDTH,
                "corridor_length": CORRIDOR_LENGTH,
                "estimated_d_min": ESTIMATED_D_MIN,
            },
        )


@configclass
class NarrowGaitRecoveryEnvCfg_PLAY(NarrowGaitRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False
