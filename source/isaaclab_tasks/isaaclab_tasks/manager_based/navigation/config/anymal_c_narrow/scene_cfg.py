from __future__ import annotations

import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# 注意：
# 如果你本地别的 anymal 配置文件里不是这个导入路径，
# 就把这里改成“和你本地已有 anymal 配置一致”的那种写法。
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


@configclass
class NarrowCorridorSceneCfg(InteractiveSceneCfg):
    """ANYmal-C narrow corridor scene for manager-based navigation."""

    # ====== global scene ======
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

    # ====== robot ======
    # 名字必须叫 robot，因为导航任务和 action 配置都按这个名字引用
    robot = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # 把机器人初始位置放在走廊中间附近
    robot.init_state.pos = (0.0, 0.0, 0.60)

    # ====== corridor geometry ======
    # 走廊沿 x 轴展开，中心在 x=0，这样和默认 reset / command 更匹配
    corridor_length = 8.0
    CORRIDOR_WIDTH: float = 0.90
    wall_height = 1.2
    wall_thickness = 0.08

    _half_len = corridor_length / 2.0
    _half_gap = corridor_width / 2.0
    _half_wall = wall_thickness / 2.0

    left_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corridor/LeftWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -(_half_gap + _half_wall), wall_height / 2.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(corridor_length, wall_thickness, wall_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.72)
            ),
        ),
    )

    right_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corridor/RightWall",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, +(_half_gap + _half_wall), wall_height / 2.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(corridor_length, wall_thickness, wall_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.70, 0.70, 0.72)
            ),
        ),
    )

    start_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corridor/StartMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-_half_len + 0.05, 0.0, 0.01),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, corridor_width, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.80, 0.20)
            ),
        ),
    )

    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Corridor/GoalMarker",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(+_half_len - 0.05, 0.0, 0.01),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, corridor_width, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.90, 0.20, 0.20)
            ),
        ),
    )

    # ====== sensors ======
    # 名字必须叫 contact_forces，因为 termination 里按这个名字引用
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
    )

    # 这个导航配置的 __post_init__ 会检查 height_scanner
    # 没有就显式设成 None，避免属性不存在
    height_scanner = None
