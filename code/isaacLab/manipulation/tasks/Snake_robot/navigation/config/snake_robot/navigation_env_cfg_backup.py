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
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.snake_robot.flat_env_cfg import SnakeRobotFlatEnvCfg

LOW_LEVEL_ENV_CFG = SnakeRobotFlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0.1), "y": (0, 0.1), "yaw": (3.14/6, 3.14/3)},
            "velocity_range": {
                "x": (-0.0, 0.0),
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
    joint_velocity_action: mdp.JointVelocityActionCfg = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "joint_1_1", "joint_4", "joint_9", "joint_14", "joint_19", "joint_24", "joint_29", "joint_34", 
            "joint_1_2", "joint_5", "joint_10", "joint_15", "joint_20", "joint_25", "joint_30"
        ],  # 关节名称
        amplitude=4.0,  # 正弦波的振幅为 2.0
        frequency=1.0,  # 正弦波的频率为 0.5Hz
        phase=torch.tensor(np.pi / 4),
        #phase=[
         #   0.0, 2*np.pi/8, 2*2*np.pi/8, 2*3*np.pi/8, 2*4*np.pi/8, 2*5*np.pi/8, 2*6*np.pi/8, 2*7*np.pi/8, 
          #  0.0, np.pi/7, 2*np.pi/7, 3*np.pi/7, 4*np.pi/7, 5*np.pi/7, 6*np.pi/7
        #],  # 每个关节不同的相位偏移
        #phase=[
         #   0.0, 4*np.pi/8, 4*2*np.pi/8, 4*3*np.pi/8, 4*4*np.pi/8, 4*5*np.pi/8, 4*6*np.pi/8, 4*7*np.pi/8, 
          #  0.0, 4*np.pi/7, 4*2*np.pi/7, 4*3*np.pi/7, 4*4*np.pi/7, 4*5*np.pi/7, 4*6*np.pi/7 
        #phase=[
          #  0.0, 2*np.pi/8, 2*2*np.pi/8, 2*3*np.pi/8, 2*4*np.pi/8, 2*5*np.pi/8, 2*6*np.pi/8, 2*7*np.pi/8, 
          #  0.0, np.pi/7, 2*np.pi/7, 3*np.pi/7, 4*np.pi/7, 5*np.pi/7, 6*np.pi/7
        #],
        #],  # 每个关节不同的相位偏移
        sine_offset=0.0,  # 无速度偏移
        use_default_offset=True  # 不使用默认关节速度偏移
    ) 
    


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )

        # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names='base_link'), "command_name": "pose_command"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh_backup,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names='base_link'), "std": 0.1, "command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(16.0, 16.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(0.3, 0.4), pos_y=(-0.3, -0.2)),
    )
    #ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-1, 1), pos_y=(-1, 1), heading=(-math.pi, math.pi)),

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


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


class NavigationEnvCfg_PLAY(NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
