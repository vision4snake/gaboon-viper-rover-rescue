# Copyright (c) 2022-2024, The Isaac Lab Project Developers.  
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Snake Robot."""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

## 
# Configuration
##

# 在此示例中，设置每个关节的力矩和速度限制

SNAKE_PR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/SnakePR",
    spawn=sim_utils.UsdFileCfg(
        #usd_path="D:\\isaacsim\\IsaacLab-main\\isaacLab.manipulation\\isaacLab\\manipulation\\assets\\usd\\snake_robot.usd",  # 这里指定蛇形机器人的USD路径
        usd_path="D:\\isaac\\IsaacLab-main\\isaacLab.manipulation\\isaacLab\\manipulation\\assets\\usd\\snake_pr.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            fix_root_link=False  # 蛇形机器人根部通常是自由的
        ),
        copy_from_source=False,
        activate_contact_sensors=True
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.08),  # 起始位置，Z轴高度可以根据需要调整 #0.15 #0.12
        # joint_pos={ 
        #     "joint_5": 0.0,
        #     "joint_10": 0.0,
        #     "joint_15": 0.0,
        #     "joint_20": 0.0,
        #     "joint_25": 0.0,
        #     "joint_30": 0.0,
        #     "joint_4": 0.0,
        #     "joint_9": 0.0,
        #     "joint_14": 0.0,
        #     "joint_19": 0.0,
        #     "joint_24": 0.0,
        #     "joint_29": 0.0,
        #     "joint_34": 0.0,
        # },
        
        joint_pos={ 
            "p1": 0.0,
            "r1": 0.0,
            "p2": 0.0,
            "r2": 0.0,
            "p3": 0.0,
            "r3": 0.0,
            "p4": 0.0,
            "r4": 0.0,
            "p5": 0.0,
            "r5": 0.0,
            "p6": 0.0,
            "r6": 0.0,
            "p7": 0.0,
            "last": 0.0,
        },
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            # joint_names_expr=[
            #     "joint_4", "joint_5", "joint_9", "joint_10", "joint_14", "joint_15", "joint_19", "joint_20", "joint_24", "joint_25", "joint_29", "joint_30", "joint_34"
            # ],
            joint_names_expr=[
              "p1", "r1", "p2", "r2", "p3", "r3", "p4", "r4", "p5", "r5",  "p6", "r6", "p7", "last"
            ],
            # stiffness=2.0,  # 刚度设置，可以根据需要调整
            # damping=1.5,     # 阻尼设置
            # effort_limit=5.0,  # 设置每个关节的最大力矩限制
            # velocity_limit=5.0,  # 设置每个关节的最大速度限制（可以根据需要调整）
            # friction=0.8
            stiffness=1.5,  # 刚度设置，可以根据需要调整
            damping=0.5,     # 阻尼设置
            effort_limit=5.0,  # 设置每个关节的最大力矩限制
            velocity_limit=5.0,  # 设置每个关节的最大速度限制（可以根据需要调整）
            friction=2.0
        ),
    },
)
