# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import torch

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.snake_robot.flat_env_cfg import SnakeRobotFlatEnvCfg

LOW_LEVEL_ENV_CFG = SnakeRobotFlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={              #  0.6, 0.6        #-0.4, -0.4
            "pose_range": {"x": (-0.2, -0.2), "y": (-0.4, -0.4), "yaw": (3.14/4 + 3.14/2, 3.14 * 3/4 + 3.14/2)}, #0, 3.14/2 #0, 0 #0, 3.14 #-0.4, -0.4
            "velocity_range": {                                    #"yaw": (3.14/4 + 3.14/2, 3.14 * 3/4 + 3.14/2) 
                "x": (-0.0, 0.0),                                   #"yaw": (3.14/6 + 3.14/2, 3.14 * 3/4 + 3.14/4)}
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    # 配置 JointVelocityActionCfg，基于正弦波的关节速度控制，并包含相位偏移
    joint_velocity_action: mdp.JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "joint_8", "joint_9", "joint_10", "joint_11", "joint_12", "joint_13"
        ],  # 关节名称
        use_default_offset=True  # 不使用默认关节速度偏移
    )
    


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        #projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        root_pos_w = ObsTerm(func=mdp.root_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        #joint_pos = ObsTerm(func=mdp.joint_pos)
        #body_incoming_wrench = ObsTerm(
        #    func=mdp.body_incoming_wrench,
        #    params={'asset_cfg': SceneEntityCfg("robot")},
        #)
        #imu_orientation = ObsTerm(
        #    func=mdp.imu_orientation,
        #)
        #imu_ang_vel = ObsTerm(
        #    func=mdp.imu_ang_vel,
        #)
        #imu_lin_acc = ObsTerm(
        #    func=mdp.imu_lin_acc,
        #)


    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)

    reward_cut_point = RewTerm(
        func=mdp.reward_cut_point,
        weight=0.3,
        params={"command_name": "pose_command"},
    )
    reward_orbit = RewTerm(
        func=mdp.reward_orbit,
        weight=0.5,
        params={"command_name": "pose_command"},
    )
    reward_heading = RewTerm(
        func=mdp.reward_heading,
        weight=0.3,
        params={"command_name": "pose_command"},
    )
    # reward_straight= RewTerm(
    #     func=mdp.reward_straight_locking,
    #     weight=0.2,
    #     params={"command_name": "pose_command"},
    # )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(90.0, 90.0), #33.0 #49.0, 49.0 #100.0, 100.0 #72.0, 72.0 #20260510 test_success 90.0, 90.0
        debug_vis=True,
        #ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-1.2, -1.0), pos_y=(-0.9, 0.1), heading=(-math.pi, math.pi)),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-0.9, -0.7), pos_y=(-0.7, -0.1), heading=(-math.pi, math.pi)),
    )

    #20260510 test_success ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-0.9, -0.7), pos_y=(-0.7, -0.1), heading=(-math.pi, math.pi)),

    #20260424 test ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-1.2, -1.0), pos_y=(-0.9, 0.1), heading=(-math.pi, math.pi)),
    #ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-1.2, -1.0), pos_y=(0.1, 0.3), heading=(-math.pi, math.pi)),

    #ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-1.2, -1.0), pos_y=(-0.9, 0.1), heading=(-math.pi, math.pi)),

    #ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-0.5, 0.5), pos_y=(0, 0.5), heading=(-math.pi, math.pi))

    #ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-1, 1), pos_y=(-1, 1), heading=(-math.pi, math.pi)),
    #ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(0.6, 0.8), pos_y=(0.6, 0.8), heading=(-math.pi, math.pi)) 
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass

    # # 1. 削弱“冲向中心”的欲望
    # deemphasize_heading = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "reward_heading",
    #         "weight": 0.05,
    #         "num_steps": 1000,
    #     },
    # )

    # # 2. 削弱“切点/趋近”奖励
    # deemphasize_cut_point = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "reward_cut_point",
    #         "weight": 0.05,
    #         "num_steps": 1000,
    #     },
    # )

    # # 3. 削弱“直线锁定”奖励（防止它僵硬地走直线而无法转弯）
    # deemphasize_straight = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "reward_straight",
    #         "weight": 0.05,
    #         "num_steps": 1000,
    #     },
    # )

    # # 4. 维持或增强 Orbit 奖励（作为对比，让它的主导地位更加突出）
    # maintain_orbit = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "reward_orbit",
    #         "weight": 0.8, # 保持原重或适当调高，比如 0.8
    #         "num_steps": 1000,
    #     },
    # )
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #base_contact = DoneTerm(
    #    func=mdp.illegal_contact,
     #   params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    #)
    bad_orientation = DoneTerm(func = mdp.bad_orientation, params= {"limit_angle": 3.14/2})


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt

 

class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 5 #10 #3
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = 'D:\\isaac\\srb_assets-main\\srb_assets-main\\srb_assets-main\\scripts\\blender\\outputs_eval2\\combined.usd'
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        #self.scene.imu.update_period = self.sim.dt
