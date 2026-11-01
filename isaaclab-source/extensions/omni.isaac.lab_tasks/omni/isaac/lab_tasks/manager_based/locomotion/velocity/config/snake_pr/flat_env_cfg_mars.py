# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import trimesh

import numpy as np

from omni.isaac.lab.utils import configclass

from .rough_env_cfg import SnakeRobotRoughEnvCfg

from omni.isaac.lab.terrains import TerrainGeneratorCfg,SubTerrainBaseCfg

import trimesh
import numpy as np

import numpy as np
import trimesh

def generate_random_height_field(size: tuple[float, float], scale: float = 0.1) -> trimesh.Trimesh:
    width, length = int(size[0] / scale), int(size[1] / scale)

    # 生成随机高度场
    heights = np.random.rand(width, length) * 0.5  # 高度范围可以根据需要调整

    # 创建网格顶点
    x = np.linspace(0, size[0], width)
    y = np.linspace(0, size[1], length)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    vertices = np.column_stack([xv.ravel(), yv.ravel(), heights.ravel()]).astype(np.float64)

    # 创建三角形索引
    faces = []
    for i in range(width - 1):
        for j in range(length - 1):
            idx = i * length + j
            faces.append([idx, idx + 1, idx + length])
            faces.append([idx + 1, idx + length + 1, idx + length])
    faces = np.array(faces, dtype=np.int32)  # 确保 faces 的索引是整数类型

    # 创建 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def mars_terrain_function(difficulty: float, cfg: SubTerrainBaseCfg):
    # 使用不同参数生成“火星地形”的高度和粗糙度
    # 此函数应返回一个包含地形网格列表和原点的元组
    
    # 示例地形网格生成代码
    terrain_mesh = generate_random_height_field(cfg.size)
    
    origin = np.array([0.0, 0.0, 0.0])

    return [terrain_mesh], origin

mars_sub_terrain_cfg = SubTerrainBaseCfg(
    function=mars_terrain_function,  # 指定用于生成火星地形的函数
    proportion=1.0,  # 生成整个地形
    size=(4, 5)  # 每个子地形的大小
)

@configclass
class SnakeRobotFlatEnvCfg(SnakeRobotRoughEnvCfg):
     def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = 'D:\\isaac\\srb_assets-main\\srb_assets-main\\srb_assets-main\\scripts\\blender\\outputs_train2\\combined.usd'
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class SnakeRobotFlatEnvCfg_PLAY(SnakeRobotFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
