# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared low-level policy config for two-layer narrow-passage tasks."""

import os

from .narrow_gait_env_cfg import NarrowGaitEnvCfg


LOW_LEVEL_ENV_CFG = NarrowGaitEnvCfg()
"""Low-level gait task config used by the high-level decision wrapper."""


LOW_LEVEL_POLICY_PATH = os.environ.get(
    "ISAAC_NARROW_LOW_LEVEL_POLICY_PATH",
    "logs/rsl_rl/anymal_c_narrow_gait/"
    "2026-07-02_16-17-04_staged_memory_near_wall_clean_v1/exported/policy.pt",
)
"""TorchScript low-level gait policy path.

The high-level navigation action wrapper loads a JIT module, not a raw RSL-RL
``model_*.pt`` checkpoint. Export a checkpoint with ``rsl_rl/play.py`` or set
``ISAAC_NARROW_LOW_LEVEL_POLICY_PATH`` to an exported ``policy.pt``.
"""
