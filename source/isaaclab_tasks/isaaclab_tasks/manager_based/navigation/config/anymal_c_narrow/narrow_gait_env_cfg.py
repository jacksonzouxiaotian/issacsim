# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level narrow-passage gait task for ANYmal-C.

Unlike the navigation configs in this package, this task trains the gait policy
directly. The action is joint position targets, not a velocity command sent to a
pre-trained locomotion policy.
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

        # Add narrow-passage geometry and short failure memory to the gait policy observation.
        self.observations.policy.corridor_state = ObsTerm(
            func=narrow_mdp.corridor_state,
            params={
                "corridor_width": CORRIDOR_WIDTH,
                "corridor_length": CORRIDOR_LENGTH,
                "estimated_d_min": ESTIMATED_D_MIN,
            },
        )
        self.observations.policy.recovery_memory = ObsTerm(
            func=narrow_mdp.recovery_memory_state,
            params={
                "corridor_width": CORRIDOR_WIDTH,
                "estimated_d_min": ESTIMATED_D_MIN,
            },
        )
        self.observations.policy.enable_corruption = True
        self.observations.policy.base_lin_vel.noise = Unoise(n_min=-0.08, n_max=0.08)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.15, n_max=0.15)

        # Retune locomotion rewards for narrow-passage gait rather than generic velocity tracking.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.feet_air_time.weight = 0.35
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.undesired_contacts.weight = -2.0

        self.rewards.forward_progress = RewTerm(func=narrow_mdp.forward_progress_reward, weight=2.5)
        self.rewards.centerline_penalty = RewTerm(
            func=narrow_mdp.centerline_error_l1,
            weight=-2.0,
            params={"corridor_width": CORRIDOR_WIDTH},
        )
        self.rewards.unsafe_clearance = RewTerm(
            func=narrow_mdp.unsafe_clearance_penalty,
            weight=-2.0,
            params={"corridor_width": CORRIDOR_WIDTH, "safety_margin": 0.08},
        )
        self.rewards.stuck_penalty = RewTerm(
            func=narrow_mdp.stuck_penalty,
            weight=-4.0,
            params={"min_forward_speed": 0.04, "goal_x": GOAL_X},
        )
        self.rewards.recovery_progress = RewTerm(
            func=narrow_mdp.recovery_progress_reward,
            weight=2.0,
            params={"min_stuck_steps": 5.0},
        )
        self.rewards.success_bonus = RewTerm(
            func=narrow_mdp.goal_reached_bonus,
            weight=180.0,
            params={
                "goal_x": GOAL_X,
                "tol": GOAL_TOL,
                "corridor_width": CORRIDOR_WIDTH,
                "lateral_margin": 0.08,
            },
        )

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
