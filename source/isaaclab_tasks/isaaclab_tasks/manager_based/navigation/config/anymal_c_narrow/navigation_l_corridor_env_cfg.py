# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""L-shaped narrow-corridor navigation environment config.

This environment extends the original straight narrow-passage task to an
L-shaped corridor. It is useful for verifying that the policy is not simply
moving forward, but must re-orient and follow a turning passage.

This version is a simplified first-stage training config:
- shorter L corridor
- wider corridor
- easier goal tolerance
- stronger path-progress reward
- direct prim paths under ENV_REGEX_NS
"""

import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.navigation.mdp as mdp
from . import mdp_narrow as narrow_mdp

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import (
    AnymalCFlatEnvCfg,
)

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


# =============================================================================
# L-corridor parameters
# =============================================================================

# First-stage easier L corridor.
# After this works, gradually increase L1/L2 to 3.0, 4.0, 5.0.
CORRIDOR_WIDTH = 1.00
CORRIDOR_L1 = 2.5
CORRIDOR_L2 = 2.5
CORRIDOR_TOTAL_LENGTH = CORRIDOR_L1 + CORRIDOR_L2

CORRIDOR_WALL_HEIGHT = 1.2
CORRIDOR_WALL_THICKNESS = 0.08

# Leave some space at the corner to avoid blocking the turn.
CORNER_OPEN_MARGIN = CORRIDOR_WIDTH * 0.55

# Segment-1 walls.
X_UPPER_WALL_LEN = CORRIDOR_L1 - CORNER_OPEN_MARGIN
X_LOWER_WALL_LEN = CORRIDOR_L1

# Segment-2 walls.
Y_LEFT_WALL_LEN = CORRIDOR_L2
Y_RIGHT_WALL_LEN = CORRIDOR_L2

# Goal near the end of the second branch.
GOAL_X = CORRIDOR_L1
GOAL_Y = CORRIDOR_L2 - 0.20

# First-stage easier tolerance.
# After learning succeeds, reduce to 0.35 then 0.25.
GOAL_TOL = 0.50

# Conservative local feasibility estimate used for Delta D = width - D_min.
ESTIMATED_D_MIN = 0.72

# Start pose.
START_X_RANGE = (-0.8, -0.4)
START_Y_RANGE = (-0.04, 0.04)
START_YAW_RANGE = (-0.10, 0.10)


# =============================================================================
# Scene
# =============================================================================

@configclass
class NarrowCorridorSceneCfg(InteractiveSceneCfg):
    """Scene with ANYmal-C traversing an L-shaped narrow corridor."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    robot = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # -------------------------------------------------------------------------
    # Segment 1: along +x
    # Centerline: y = 0, x in [0, CORRIDOR_L1]
    # -------------------------------------------------------------------------

    # Upper wall of the first branch.
    # It ends before the corner to leave turning space.
    x_upper_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/XUpperWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                X_UPPER_WALL_LEN / 2.0,
                +(CORRIDOR_WIDTH / 2.0 + CORRIDOR_WALL_THICKNESS / 2.0),
                CORRIDOR_WALL_HEIGHT / 2.0,
            ),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(
                X_UPPER_WALL_LEN,
                CORRIDOR_WALL_THICKNESS,
                CORRIDOR_WALL_HEIGHT,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.72)
            ),
        ),
    )

    # Lower wall of the first branch.
    # This prevents the robot from escaping downward.
    x_lower_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/XLowerWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                X_LOWER_WALL_LEN / 2.0,
                -(CORRIDOR_WIDTH / 2.0 + CORRIDOR_WALL_THICKNESS / 2.0),
                CORRIDOR_WALL_HEIGHT / 2.0,
            ),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(
                X_LOWER_WALL_LEN,
                CORRIDOR_WALL_THICKNESS,
                CORRIDOR_WALL_HEIGHT,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.72)
            ),
        ),
    )

    # -------------------------------------------------------------------------
    # Segment 2: along +y
    # Centerline: x = CORRIDOR_L1, y in [0, CORRIDOR_L2]
    # -------------------------------------------------------------------------

    # Left wall of the second branch.
    y_left_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/YLeftWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                CORRIDOR_L1 - (CORRIDOR_WIDTH / 2.0 + CORRIDOR_WALL_THICKNESS / 2.0),
                Y_LEFT_WALL_LEN / 2.0,
                CORRIDOR_WALL_HEIGHT / 2.0,
            ),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(
                CORRIDOR_WALL_THICKNESS,
                Y_LEFT_WALL_LEN,
                CORRIDOR_WALL_HEIGHT,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.72)
            ),
        ),
    )

    # Right wall of the second branch.
    y_right_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/YRightWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                CORRIDOR_L1 + (CORRIDOR_WIDTH / 2.0 + CORRIDOR_WALL_THICKNESS / 2.0),
                Y_RIGHT_WALL_LEN / 2.0,
                CORRIDOR_WALL_HEIGHT / 2.0,
            ),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(
                CORRIDOR_WALL_THICKNESS,
                Y_RIGHT_WALL_LEN,
                CORRIDOR_WALL_HEIGHT,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.72)
            ),
        ),
    )

    # Start marker.
    start_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StartMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.01),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, CORRIDOR_WIDTH * 0.85, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.80, 0.20)
            ),
        ),
    )

    # Corner marker, useful for visual debugging.
    corner_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CornerMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_L1, 0.0, 0.012),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.40, 0.90)
            ),
        ),
    )

    # Goal marker.
    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(GOAL_X, GOAL_Y, 0.01),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_WIDTH * 0.85, 0.03, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.90, 0.20, 0.20)
            ),
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )

    # Disable height scanner for this first version.
    height_scanner = None


# =============================================================================
# Events
# =============================================================================

@configclass
class EventCfg:
    """Events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": START_X_RANGE,
                "y": START_Y_RANGE,
                "yaw": START_YAW_RANGE,
            },
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


# =============================================================================
# Actions
# =============================================================================

@configclass
class ActionsCfg:
    """Action terms."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


# =============================================================================
# Observations
# =============================================================================

@configclass
class ObservationsCfg:
    """Observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations for L-shaped narrow-passage traversal."""

        # Robot motion state.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Goal command.
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pose_command"},
        )

        # Previous action.
        last_action = ObsTerm(func=mdp.last_action)

        # L-corridor privileged geometric observation.
        # This function must exist in mdp_narrow.py.
        corridor_state = ObsTerm(
            func=narrow_mdp.l_corridor_state,
            params={
                "corridor_width": CORRIDOR_WIDTH,
                "corner_x": CORRIDOR_L1,
                "goal_y": GOAL_Y,
                "estimated_d_min": ESTIMATED_D_MIN,
            },
        )

        recovery_memory = ObsTerm(
            func=narrow_mdp.recovery_memory_state,
            params={
                "corridor_width": CORRIDOR_WIDTH,
                "estimated_d_min": ESTIMATED_D_MIN,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Rewards
# =============================================================================

@configclass
class RewardsCfg:
    """Rewards specialized for L-shaped narrow-corridor traversal."""

    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.05,
    )

    termination_penalty = RewTerm(
        func=narrow_mdp.failure_termination_penalty,
        weight=-300.0,
        params={"term_names": ("base_contact", "bad_orientation", "base_too_low")},
    )

    # Strengthened position tracking.
    # The original 0.05 was too weak and often became nearly useless.
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "std": 2.0,
            "command_name": "pose_command",
        },
    )

    # Keep heading reward mild for the first stage.
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.02,
        params={"command_name": "pose_command"},
    )

    # Main reward: progress along L-shaped path.
    # The function should reward x progress before corner and y progress after corner.
    forward_progress = RewTerm(
        func=narrow_mdp.l_corridor_path_progress,
        weight=6.0,
        params={
            "corner_x": CORRIDOR_L1,
            "goal_y": GOAL_Y,
            "corridor_width": CORRIDOR_WIDTH,
        },
    )

    path_progress_dense = RewTerm(
        func=narrow_mdp.l_corridor_progress_dense,
        weight=4.0,
        params={
            "corner_x": CORRIDOR_L1,
            "goal_y": GOAL_Y,
            "corridor_width": CORRIDOR_WIDTH,
        },
    )

    goal_distance = RewTerm(
        func=narrow_mdp.l_corridor_goal_distance_tanh,
        weight=3.0,
        params={
            "goal_x": GOAL_X,
            "goal_y": GOAL_Y,
            "std": 1.5,
        },
    )

    corner_bonus = RewTerm(
        func=narrow_mdp.l_corridor_corner_reached_bonus,
        weight=20.0,
        params={
            "corner_x": CORRIDOR_L1,
            "tol": 0.40,
        },
    )

    # Do not make this too strong at the beginning, otherwise policy may freeze.
    centerline_penalty = RewTerm(
        func=narrow_mdp.l_corridor_centerline_error,
        weight=-1.0,
        params={
            "corner_x": CORRIDOR_L1,
            "corridor_width": CORRIDOR_WIDTH,
        },
    )

    unsafe_clearance = RewTerm(
        func=narrow_mdp.unsafe_clearance_penalty,
        weight=-1.5,
        params={
            "corridor_width": CORRIDOR_WIDTH,
            "safety_margin": 0.08,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*HIP.*", ".*THIGH.*"],
            ),
            "threshold": 1.5,
        },
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.03,
    )

    recovery_progress = RewTerm(
        func=narrow_mdp.recovery_progress_reward,
        weight=3.0,
        params={"min_stuck_steps": 5.0},
    )

    oscillation_penalty = RewTerm(
        func=narrow_mdp.oscillation_penalty,
        weight=-0.35,
    )

    # Small time penalty to encourage completion.
    time_penalty = RewTerm(
        func=narrow_mdp.unfinished_time_penalty_xy,
        weight=-0.03,
        params={
            "goal_x": GOAL_X,
            "goal_y": GOAL_Y,
            "tol": GOAL_TOL,
        },
    )

    success_bonus = RewTerm(
        func=narrow_mdp.goal_reached_bonus_xy,
        weight=250.0,
        params={
            "goal_x": GOAL_X,
            "goal_y": GOAL_Y,
            "tol": GOAL_TOL,
        },
    )


# =============================================================================
# Commands
# =============================================================================

@configclass
class CommandsCfg:
    """Commands."""

    # Goal is located near the end of the vertical branch.
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(GOAL_X - 0.10, GOAL_X + 0.10),
            pos_y=(GOAL_Y - 0.10, GOAL_Y + 0.10),
            heading=(math.pi / 2.0 - 0.20, math.pi / 2.0 + 0.20),
        ),
    )


# =============================================================================
# Terminations
# =============================================================================

@configclass
class TerminationsCfg:
    """Terminations for L-shaped narrow-corridor traversal."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    goal_reached = DoneTerm(
        func=narrow_mdp.goal_reached_xy,
        params={
            "goal_x": GOAL_X,
            "goal_y": GOAL_Y,
            "tol": GOAL_TOL,
        },
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 6.0,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )

    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.28},
    )


# =============================================================================
# Env
# =============================================================================

@configclass
class NavigationNarrowEnvCfg(ManagerBasedRLEnvCfg):
    """L-shaped narrow corridor navigation environment."""

    scene = NarrowCorridorSceneCfg(
        num_envs=LOW_LEVEL_ENV_CFG.scene.num_envs,
        env_spacing=8.0,
    )

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt

        # High-level policy is slower than the low-level locomotion controller.
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.sim.render_interval = self.decimation

        # First-stage L corridor is short, 18s is enough.
        self.episode_length_s = 18.0

        # Start before the entrance.
        self.scene.robot.init_state.pos = (-0.6, 0.0, 0.6)

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class NavigationNarrowEnvCfg_PLAY(NavigationNarrowEnvCfg):
    """Play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 8.0
        self.observations.policy.enable_corruption = False

        print("=" * 80)
        print("[L-CORRIDOR PLAY CONFIG]")
        print(f"CORRIDOR_WIDTH        = {CORRIDOR_WIDTH}")
        print(f"CORRIDOR_L1           = {CORRIDOR_L1}")
        print(f"CORRIDOR_L2           = {CORRIDOR_L2}")
        print(f"GOAL_X, GOAL_Y        = ({GOAL_X}, {GOAL_Y})")
        print(f"GOAL_TOL              = {GOAL_TOL}")
        print(f"X_UPPER_WALL_LEN      = {X_UPPER_WALL_LEN}")
        print(f"X_LOWER_WALL_LEN      = {X_LOWER_WALL_LEN}")
        print("=" * 80)


def get_corridor_dimensions():
    return {
        "type": "L-corridor-stage-1",
        "width": CORRIDOR_WIDTH,
        "l1": CORRIDOR_L1,
        "l2": CORRIDOR_L2,
        "total_length": CORRIDOR_TOTAL_LENGTH,
        "wall_height": CORRIDOR_WALL_HEIGHT,
        "wall_thickness": CORRIDOR_WALL_THICKNESS,
        "goal_x": GOAL_X,
        "goal_y": GOAL_Y,
        "goal_tol": GOAL_TOL,
    }
