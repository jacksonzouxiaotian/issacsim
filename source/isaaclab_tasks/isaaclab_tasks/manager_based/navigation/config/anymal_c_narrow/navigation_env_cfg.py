# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Narrow-corridor navigation environment config (modified for narrow-passage traversal)."""

import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.navigation.mdp as mdp
from . import mdp_narrow as narrow_mdp # <<< 新增的窄通道辅助模块
from .low_level_policy_cfg import LOW_LEVEL_ENV_CFG, LOW_LEVEL_POLICY_PATH

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

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


# =============================================================================
# Corridor parameters
# =============================================================================
CORRIDOR_WIDTH = 0.85          # 通道宽度
CORRIDOR_LENGTH = 5.0          # 局部窄通道长度
CORRIDOR_WALL_HEIGHT = 1.2     # 通道墙高度
CORRIDOR_WALL_THICKNESS = 0.08  # 通道墙厚度

# 这个 margin 是“命令采样 / 目标点”离墙的安全边界，不是理论极限
CORRIDOR_COMMAND_MARGIN = 0.05

# 成功判定
GOAL_X = CORRIDOR_LENGTH - 0.20
GOAL_TOL = 0.20

# Conservative local feasibility estimate used for Delta D = width - D_min.
ESTIMATED_D_MIN = 0.72


# =============================================================================
# Scene
# =============================================================================
@configclass
class NarrowCorridorSceneCfg(InteractiveSceneCfg):
    """Scene with ANYmal-C traversing a straight narrow corridor."""

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

    left_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LeftWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                CORRIDOR_LENGTH / 2.0,
                -(CORRIDOR_WIDTH / 2.0 + CORRIDOR_WALL_THICKNESS / 2.0),
                CORRIDOR_WALL_HEIGHT / 2.0,
            ),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS, CORRIDOR_WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.72)
            ),
        ),
    )

    right_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RightWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                CORRIDOR_LENGTH / 2.0,
                +(CORRIDOR_WIDTH / 2.0 + CORRIDOR_WALL_THICKNESS / 2.0),
                CORRIDOR_WALL_HEIGHT / 2.0,
            ),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS, CORRIDOR_WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.72)
            ),
        ),
    )

    start_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StartMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.01),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, CORRIDOR_WIDTH, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.80, 0.20)
            ),
        ),
    )

    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CORRIDOR_LENGTH, 0.0, 0.01),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, CORRIDOR_WIDTH, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.90, 0.20, 0.20)
            ),
        ),
    )

    # 用接触传感器，但不要只拿来“立刻失败”，也要参与奖励
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )

    height_scanner = None


# =============================================================================
# Events
# =============================================================================
@configclass
class EventCfg:
    """Events."""

    # 从走廊入口外侧开始，而不是一开始就在走廊中间
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, -0.4),     # 入口外
                "y": (-0.03, 0.03),
                "yaw": (-0.08, 0.08),
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
        policy_path=LOW_LEVEL_POLICY_PATH,
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
        """Policy observations for narrow-passage traversal."""

        # 运动状态
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # 高层命令
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pose_command"},
        )

        # 上一步动作，减少抖动
        last_action = ObsTerm(func=mdp.last_action)

        # 窄通道专用“特权几何观测”
        corridor_state = ObsTerm(
            func=narrow_mdp.corridor_state,
            params={
                "corridor_width": CORRIDOR_WIDTH,
                "corridor_length": CORRIDOR_LENGTH,
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
    """Rewards specialized for narrow-corridor traversal."""

    alive = RewTerm(func=mdp.is_alive, weight=0.05)

    termination_penalty = RewTerm(
        func=narrow_mdp.failure_termination_penalty,
        weight=-300.0,
        params={"term_names": ("stuck", "base_contact", "bad_orientation", "base_too_low")},
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1.5,
        params={"std": 1.2, "command_name": "pose_command"},
    )

    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.03,
        params={"command_name": "pose_command"},
    )

    forward_progress = RewTerm(
        func=narrow_mdp.forward_progress_reward,
        weight=5.0,
    )

    centerline_penalty = RewTerm(
        func=narrow_mdp.centerline_error_l1,
        weight=-2.5,
        params={"corridor_width": CORRIDOR_WIDTH},
    )

    unsafe_clearance = RewTerm(
        func=narrow_mdp.unsafe_clearance_penalty,
        weight=-2.0,
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
                body_names=["base", ".*HIP.*", ".*THIGH.*"]
            ),
            "threshold": 1.5,
        },
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

    stuck_penalty = RewTerm(
        func=narrow_mdp.stuck_penalty,
        weight=-6.0,
        params={
            "min_forward_speed": 0.04,
            "goal_x": GOAL_X,
        },
    )

    recovery_progress = RewTerm(
        func=narrow_mdp.recovery_progress_reward,
        weight=3.0,
        params={"min_stuck_steps": 5.0},
    )

    oscillation_penalty = RewTerm(
        func=narrow_mdp.oscillation_penalty,
        weight=-0.4,
    )

    # 没到终点前，每一步都扣一点，防止磨时间
    time_penalty = RewTerm(
        func=narrow_mdp.unfinished_time_penalty,
        weight=-0.05,
        params={
            "goal_x": GOAL_X,
        },
    )

    success_bonus = RewTerm(
        func=narrow_mdp.clean_goal_reached_bonus,
        weight=220.0,
        params={
            "goal_x": GOAL_X,
            "tol": GOAL_TOL,
            "corridor_width": CORRIDOR_WIDTH,
            "lateral_margin": 0.08,
        },
    )


# =============================================================================
# Commands
# =============================================================================
@configclass
class CommandsCfg:
    """Commands."""

    # 目标尽量固定在出口附近，先学“稳定穿过去”
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(CORRIDOR_LENGTH - 0.5, CORRIDOR_LENGTH - 0.2),
            pos_y=(
                -CORRIDOR_COMMAND_MARGIN,
                +CORRIDOR_COMMAND_MARGIN,
            ),
            heading=(-0.06, 0.06),
        ),
    )


# =============================================================================
# Terminations
# =============================================================================
@configclass
class TerminationsCfg:
    """Terminations for narrow-corridor traversal."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 到出口就立刻成功结束
    goal_reached = DoneTerm(
        func=narrow_mdp.goal_reached,
        params={
            "goal_x": GOAL_X,
            "tol": GOAL_TOL,
            "corridor_width": CORRIDOR_WIDTH,
            "lateral_margin": 0.08,
        },
    )

    # 卡住更早结束，不给它磨时间
    stuck = DoneTerm(
        func=narrow_mdp.stuck_for_steps,
        params={
            "window_s": 1.2,
            "min_forward_speed": 0.03,
            "goal_x": GOAL_X,
        },
    )

    # 严重基座碰撞才终止
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 6.0,
        },
    )

    # 摔倒
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},
    )

    # base 太低
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.28},
    )


# =============================================================================
# Env
# =============================================================================
@configclass
class NavigationNarrowEnvCfg(ManagerBasedRLEnvCfg):
    """Narrow corridor navigation environment with narrow-passage shaping."""

    scene = NarrowCorridorSceneCfg(
        num_envs=LOW_LEVEL_ENV_CFG.scene.num_envs,
        env_spacing=14.0,   # 必须明显大于走廊长度，避免多环境互相串扰
    )

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        # 高层动作频率比低层慢一些
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.sim.render_interval = self.decimation

        # 局部通过任务不需要完整导航长度。
        self.episode_length_s = 12.0

        # 默认初始位姿也放到入口外
        self.scene.robot.init_state.pos = (-0.7, 0.0, 0.6)

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class NavigationNarrowEnvCfg_PLAY(NavigationNarrowEnvCfg):
    """Play config."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 15.0
        self.observations.policy.enable_corruption = False


def get_corridor_dimensions():
    return {
        "width": CORRIDOR_WIDTH,
        "length": CORRIDOR_LENGTH,
        "wall_height": CORRIDOR_WALL_HEIGHT,
        "wall_thickness": CORRIDOR_WALL_THICKNESS,
    }
