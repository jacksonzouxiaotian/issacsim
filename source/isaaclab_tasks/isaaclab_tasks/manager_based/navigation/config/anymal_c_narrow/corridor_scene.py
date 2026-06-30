# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom corridor scene generator for IsaacLab.

This module provides functions to generate a narrow corridor environment
for training robot navigation.
"""

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


# Corridor dimensions
CORRIDOR_WIDTH = 0.7
CORRIDOR_LENGTH = 8.0
CORRIDOR_WALL_HEIGHT = 1.2
CORRIDOR_WALL_THICKNESS = 0.08


@configclass
class CorridorWallCfg:
    """Configuration for corridor walls."""
    
    width: float = CORRIDOR_WIDTH
    length: float = CORRIDOR_LENGTH
    wall_height: float = CORRIDOR_WALL_HEIGHT
    wall_thickness: float = CORRIDOR_WALL_THICKNESS
    wall_color: tuple = (0.7, 0.7, 0.72)
    start_marker_color: tuple = (0.2, 0.8, 0.2)
    goal_marker_color: tuple = (0.9, 0.2, 0.2)


def create_corridor_walls(stage: Usd.UsdStage, prim_path: str = "/World/Corridor") -> None:
    """Create corridor walls in the USD stage.
    
    Args:
        stage: The USD stage to add the corridor to.
        prim_path: The root path for corridor primitives.
    """
    # Calculate dimensions
    half_gap = CORRIDOR_WIDTH / 2.0
    half_wall_thickness = CORRIDOR_WALL_THICKNESS / 2.0
    wall_size = (CORRIDOR_LENGTH, CORRIDOR_WALL_THICKNESS, CORRIDOR_WALL_HEIGHT)
    
    # Left wall position
    left_wall_pos = Gf.Vec3f(
        CORRIDOR_LENGTH / 2.0,
        -(half_gap + half_wall_thickness),
        CORRIDOR_WALL_HEIGHT / 2.0
    )
    
    # Right wall position
    right_wall_pos = Gf.Vec3f(
        CORRIDOR_LENGTH / 2.0,
        +(half_gap + half_wall_thickness),
        CORRIDOR_WALL_HEIGHT / 2.0
    )
    
    # Create left wall
    _create_wall(
        stage=stage,
        path=f"{prim_path}/LeftWall",
        size=wall_size,
        position=left_wall_pos,
        color=CORRIDOR_WALL_CFG.wall_color,
    )
    
    # Create right wall
    _create_wall(
        stage=stage,
        path=f"{prim_path}/RightWall",
        size=wall_size,
        position=right_wall_pos,
        color=CORRIDOR_WALL_CFG.wall_color,
    )
    
    # Create start marker (green)
    start_marker_size = (0.03, CORRIDOR_WIDTH, 0.02)
    start_marker_pos = Gf.Vec3f(0.0, 0.0, 0.01)
    _create_wall(
        stage=stage,
        path=f"{prim_path}/StartMarker",
        size=start_marker_size,
        position=start_marker_pos,
        color=CORRIDOR_WALL_CFG.start_marker_color,
    )
    
    # Create goal marker (red)
    goal_marker_size = (0.03, CORRIDOR_WIDTH, 0.02)
    goal_marker_pos = Gf.Vec3f(CORRIDOR_LENGTH, 0.0, 0.01)
    _create_wall(
        stage=stage,
        path=f"{prim_path}/GoalMarker",
        size=goal_marker_size,
        position=goal_marker_pos,
        color=CORRIDOR_WALL_CFG.goal_marker_color,
    )


def _create_wall(
    stage: Usd.UsdStage,
    path: str,
    size: tuple,
    position: Gf.Vec3f,
    color: tuple,
) -> None:
    """Create a single wall or marker.
    
    Args:
        stage: The USD stage.
        path: The prim path.
        size: The size (x, y, z).
        position: The position.
        color: The RGB color (0-1).
    """
    # Create cube prim
    cube = UsdGeom.Cube.Define(stage, path)
    cube.AddTranslateOp().Set(position)
    cube.AddScaleOp().Set(Gf.Vec3f(size[0] / 2, size[1] / 2, size[2] / 2))
    
    # Add display color
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])
    
    # Add collision
    prim = stage.GetPrimAtPath(path)
    if not prim.HasAPI(UsdGeomCollisionAPI):
        collision_api = UsdGeomCollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr(True)


# For backwards compatibility
CORRIDOR_WALL_CFG = CorridorWallCfg()


def design_corridor_scene():
    """Design the corridor scene using IsaacLab's sim utilities.
    
    This function creates:
    - Ground plane
    - Lighting
    - Left and right walls
    - Start and goal markers
    """
    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    
    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/Light", light_cfg)
    
    # Corridor parameters
    corridor_width = CORRIDOR_WIDTH
    corridor_length = CORRIDOR_LENGTH
    wall_height = CORRIDOR_WALL_HEIGHT
    wall_thickness = CORRIDOR_WALL_THICKNESS
    
    # Wall size
    wall_size = (corridor_length, wall_thickness, wall_height)
    left_wall_y = -(corridor_width / 2.0 + wall_thickness / 2.0)
    right_wall_y = +(corridor_width / 2.0 + wall_thickness / 2.0)
    
    # Wall configuration
    wall_cfg = sim_utils.CuboidCfg(
        size=wall_size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CORRIDOR_WALL_CFG.wall_color),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=None,
        mass_props=None,
    )
    
    # Left wall
    wall_cfg.func(
        "/World/Corridor/LeftWall",
        wall_cfg,
        translation=(corridor_length / 2.0, left_wall_y, wall_height / 2.0),
    )
    
    # Right wall
    wall_cfg.func(
        "/World/Corridor/RightWall",
        wall_cfg,
        translation=(corridor_length / 2.0, right_wall_y, wall_height / 2.0),
    )
    
    # Start marker (green)
    start_cfg = sim_utils.CuboidCfg(
        size=(0.03, corridor_width, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CORRIDOR_WALL_CFG.start_marker_color),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=None,
        mass_props=None,
    )
    start_cfg.func(
        "/World/Corridor/StartMarker",
        start_cfg,
        translation=(0.0, 0.0, 0.01),
    )
    
    # Goal marker (red)
    goal_cfg = sim_utils.CuboidCfg(
        size=(0.03, corridor_width, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CORRIDOR_WALL_CFG.goal_marker_color),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=None,
        mass_props=None,
    )
    goal_cfg.func(
        "/World/Corridor/GoalMarker",
        goal_cfg,
        translation=(corridor_length, 0.0, 0.01),
    )


def get_corridor_bounds():
    """Return the corridor bounding box.
    
    Returns:
        dict: Bounding box with 'min' and 'max' positions.
    """
    return {
        "min": Gf.Vec3f(0, -CORRIDOR_WIDTH / 2, 0),
        "max": Gf.Vec3f(CORRIDOR_LENGTH, CORRIDOR_WIDTH / 2, CORRIDOR_WALL_HEIGHT),
    }
