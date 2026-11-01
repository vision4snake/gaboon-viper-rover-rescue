# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
import omni.isaac.lab.utils.math as math_utils

import carb
import numpy as np
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.managers import SceneEntityCfg
import time

from omni.isaac.lab.envs import ManagerBasedEnv,ManagerBasedRLEnv

from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.joint_names = cfg.joint_names
        self.start_time = time.time()  # 记录初始时间，用于计算当前时间 t
        self._asset: Articulation = env.scene[asset_cfg.name]
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = 1 #len(self._joint_ids)
        self.env = env
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        print("self._asset.num_joints:",self._asset.num_joints)
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        #self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        #self._processed_actions = torch.zeros_like(self.raw_actions)
        self._current_time = 0
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def current_time(self):
        self._current_time = torch.tensor(time.time() - self.start_time, dtype=torch.float32)
        return self._current_time

    """
    Operations.
    """

    def process_actions_2(self, actions: torch.Tensor):
        # store the raw actions
        #self._processed_actions[:] = scaled_actions[:]
        #print("actions:",actions)       
        raw_amplitude_odd = actions[:, :7].clamp(0, 1.0)  # 取第二维度的前七个值并进行clamp
        amplitude_odd = raw_amplitude_odd * 10

        #self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        #scaled_actions = self._raw_actions * 10
        #print("scaled_actions:",scaled_actions)
        amplitude_even = torch.tensor(18.0).to(self.device)

        raw_frequency_odd = actions[:, 7:14].clamp(0, 1.0)  # 提取第8到第14个值并进行clamp
        frequency_odd = raw_frequency_odd * 1.4

        frequency_even = 1.4  # 固定频率 ω

        raw_phase_odd = actions[:, 14:21].clamp(-1.0, 1.0)
        phase_odd = raw_phase_odd * torch.tensor(np.pi).to(self.device)

        # 如果 cfg.phase 是 float，则将其扩展为与关节数目相同的列表
        phase_even = torch.tensor(np.pi/4).to(self.device)

        # 计算当前时间
        current_time = torch.tensor(time.time() - self.start_time, dtype=torch.float32)
        self._current_time = current_time
        # 创建正弦波速度控制信号
        #sine_wave_velocities = torch.zeros([:,len(self.joint_names)], dtype=torch.float32)
        sine_wave_velocities = torch.zeros(self.num_envs, len(self.joint_names), device=self.device)

        #print("amplitude_even:",amplitude_even)

        # 对每个关节生成正弦波速度
        for i in range(len(self.joint_names)):
            if i % 2 == 0:  # 偶数关节
                phase_offset = (i / 2) * phase_even

                desired_velocity = (
                amplitude_even * torch.cos(torch.pi * frequency_even * current_time + phase_offset) 
                )
                #print(f"desired_velocity shape: {desired_velocity.shape}")  
                #print(f"sine_wave_velocities[i] shape: {sine_wave_velocities[i].shape}")

                sine_wave_velocities[:, i] = desired_velocity
            else:  # 奇数关节
                phase_offset = phase_odd[:,int((i-1)/2)]

                desired_velocity = (
                    amplitude_odd[:,int((i-1)/2)] * torch.cos(torch.pi * frequency_odd[:,int((i-1)/2)] * current_time + phase_offset) 
                )

                #sine_wave_velocities[:, i] = scaled_actions[:,int((i-1)/2)]
                sine_wave_velocities[:, i] = desired_velocity

        self._processed_actions[:] = sine_wave_velocities[:]


    def sigmoid_decay(self, joint_idx, max_idx, steepness=10):
        """使用Sigmoid函数平滑递减，避免突然下降"""
        joint_idx_tensor = torch.tensor(joint_idx, dtype=torch.float32)
        max_idx_tensor = torch.tensor(max_idx, dtype=torch.float32)

        return 1 / (1 + torch.exp(-steepness * (joint_idx_tensor / max_idx_tensor - 0.5)))
    
    # def process_actions(self, actions: torch.Tensor, episode_length_buf): 
    #     # 计算每个环境的policy（维度[1024]），每个元素为0、1、2
    #     #self._raw_actions[:] = actions.clamp(0, 1.0)
    #     #policy = torch.argmax(self._raw_actions, dim=1)  # shape: [1024]
    #     #print("policy:", policy.shape)
    #     # 对 actions 使用 Softmax 计算概率分布
    #     self._raw_actions[:] = torch.softmax(actions, dim=1)  # shape: [batch_size, 3]
        
    #     # 从概率分布中采样动作
    #     #action_distribution = torch.distributions.Categorical( self._raw_actions)
    #     #policy = action_distribution.sample()  # shape: [batch_size]，离散动作的采样结果
    #     #print("policy:",policy[0])
    #     policy = torch.argmax(self._raw_actions, dim=1)
        
    #     # 当前时间（假设episode_length_buf维度为[1024]，每个环境有独立时间）
    #     current_time = episode_length_buf  # shape: [1024]
    #     # 时间掩码：current_time >= 2时为1，否则为0（维度[1024]）
    #     time_mask = (current_time >= 2).float()  # shape: [1024]
        
    #     # 初始化速度张量，shape: [1024, num_joints]（num_joints是关节数量）
    #     sine_wave_velocities = torch.zeros(self.num_envs, len(self.joint_names), device=self.device)
        
    #     # 预计算常量（避免重复计算）
    #     const1 = 2 * torch.pi * 256 / 4096  # 偶数关节基础系数
    #     const2 = 2 * torch.pi * 120 / 4096  # 奇数关节正速度
    #     const3 = -const2  # 奇数关节负速度
        
    #     # 对每个关节生成速度（按关节循环，每个关节单独处理）
    #     for i in range(len(self.joint_names)):
    #         if i % 2 == 0:  # 偶数关节：三种policy下计算方式相同
    #             # 相位偏移（每个关节固定）
    #             phase_offset = i * 2 * torch.pi / 12
    #             # 计算余弦项（维度[1024]，每个环境独立计算）
    #             cos_term = torch.cos(8 * torch.pi / 4 * current_time + phase_offset)
    #             # 偶数关节速度 = 基础系数 * 余弦项 * 时间掩码（向量化操作）
    #             desired_velocity = const1 * cos_term * time_mask  # shape: [1024]
    #             # 赋值给当前关节的所有环境
    #             sine_wave_velocities[:, i] = desired_velocity
            
    #         else:  # 奇数关节：根据policy不同赋值
    #             mask0 = (policy == 0)  # policy=0的环境为True
    #             mask1 = (policy == 1)  # policy=1的环境为True
    #             mask2 = (policy == 2)  # policy=2的环境为True
                
    #             # 明确处理 policy == 0 的情况：设定 desired_velocity = 0
    #             desired_velocity = torch.zeros_like(policy, device=self.device)  # 默认设置为0
                
    #             # 向量化计算：根据掩码分配不同速度（0、const2、const3）
    #             desired_velocity[mask1] = const2  # policy=1时，速度设为const2
    #             desired_velocity[mask2] = const3  # policy=2时，速度设为const3
                
    #             # 赋值给当前关节的所有环境
    #             sine_wave_velocities[:, i] = desired_velocity
        
    #     # 存储处理后的动作
    #     self._processed_actions[:] = sine_wave_velocities[:]

    def process_actions(self, actions: torch.Tensor, episode_length_buf):
        # 计算每个环境的policy（维度[1024]），每个元素为0、1、2
        #self._raw_actions[:] = actions.clamp(0, 1.0)
        #policy = torch.argmax(self._raw_actions, dim=1)  # shape: [1024]
        #print("policy:", policy.shape)
        # 对 actions 使用 Softmax 计算概率分布
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        
        # 当前时间（假设episode_length_buf维度为[1024]，每个环境有独立时间）
        current_time = episode_length_buf  # shape: [1024]
        # 时间掩码：current_time >= 2时为1，否则为0（维度[1024]）
        time_mask = (current_time >= 2).float()  # shape: [1024]
        
        # 初始化速度张量，shape: [1024, num_joints]（num_joints是关节数量）
        sine_wave_velocities = torch.zeros(self.num_envs, len(self.joint_names), device=self.device)
        
        # 预计算常量（避免重复计算）
        const1 = 1.5 * torch.pi * 256 / 4096  # 偶数关节基础系数
        # 映射范围：action的值映射到[-2 * torch.pi * 120 / 4096, 2 * torch.pi * 120 / 4096]
        action_to_velocity_range = 2 * torch.pi * 120 / 4096  # 最大速度
        
        # 对每个关节生成速度（按关节循环，每个关节单独处理）
        for i in range(len(self.joint_names)):
            if i % 2 == 0:  # 偶数关节：三种policy下计算方式相同
                # 相位偏移（每个关节固定）
                phase_offset = i * 2 * torch.pi / 12
                # 计算余弦项（维度[1024]，每个环境独立计算）
                cos_term = torch.cos(8 * torch.pi / 4 * current_time + phase_offset)
                # 偶数关节速度 = 基础系数 * 余弦项 * 时间掩码（向量化操作）
                desired_velocity = const1 * cos_term * time_mask  # shape: [1024]
                # 赋值给当前关节的所有环境
                sine_wave_velocities[:, i] = desired_velocity
            
            else:  # 奇数关节：根据action映射速度
                # 将action映射到[-action_to_velocity_range, action_to_velocity_range]
                desired_velocity = 0#(0 * action_to_velocity_range).squeeze() * time_mask # 将action映射到对应的速度范围
                #desired_velocity = desired_velocity.squeeze()
                
                # 赋值给当前关节的所有环境
                sine_wave_velocities[:, i] = desired_velocity
        
        # 存储处理后的动作
        self._processed_actions[:] = sine_wave_velocities[:]

        #print(dir(self.env.command_manager))

        #在函数的最后记录日志
        # root_pos = (self._asset.data.root_pos_w[0] - self.env.scene.env_origins[0]).cpu().numpy()
        # pos_command = self.env.command_manager.get_command_data()
 

    def process_actions_backup(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        scaled_actions = self._raw_actions * self._scale + self._offset
        self._processed_actions[:] = scaled_actions[:]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self.processed_actions.clamp(
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    
class JointSineWavePositionAction(JointPositionAction):
    """Joint action term that applies sinusoidal position control to the articulation's joints."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.amplitude = cfg.amplitude  # 振幅 A
        self.frequency = cfg.frequency  # 频率 ω
        self.joint_names = cfg.joint_names
        self.phase = cfg.phase #if isinstance(cfg.phase, list) else [cfg.phase] * len(self._joint_ids)
        # 如果 cfg.phase 是 float，则将其扩展为与关节数目相同的列表
        self.sine_offset = cfg.sine_offset  # 偏移 θ0
        self.start_time = time.time()  # 记录初始时间，用于计算当前时间 t

    def get_joint_index(self, joint_name: str) -> int:
        """根据关节名称返回关节的索引"""
        return self.joint_names.index(joint_name)
    
    def apply_actions(self):
        # 计算当前时间
        current_time = time.time() - self.start_time

        # 创建正弦波速度控制信号
        sine_wave_positions = torch.zeros(len(self.joint_names), dtype=torch.float32)
        
        # 生成正弦波位置控制信号
        # 对每个关节生成正弦波位置
        for i in range(len(self.joint_names)):
            desired_velocity = (
                self.amplitude * torch.cos(2 * torch.pi * self.frequency * current_time + torch.tensor(self.phase[i], dtype=torch.float32)) 
                + self.sine_offset
            )
            sine_wave_positions[i] = desired_velocity
        
        #sine_wave_positions = self.amplitude * torch.sin(2 * torch.pi * self.frequency * current_time + self.phase) + self.offset

        # 应用到每个关节的目标位置
        #self.processed_actions = sine_wave_positions.expand_as(self.processed_actions)

        # 设置位置目标
        #self._asset.set_joint_velocity_target(sine_wave_velocities, joint_ids=self._joint_ids)
        self._asset.set_joint_position_target(sine_wave_positions, joint_ids=self._joint_ids)


class RelativeJointPositionAction(JointAction):
    r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.

    Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
    This means that the processed actions are added to the current joint positions of the articulation's joints
    before being sent as position commands.

    This means that the action applied at every step is:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.RelativeJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use zero offset for relative position
        if cfg.use_zero_offset:
            self._offset = 0.0

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        current_actions = current_actions.clamp(
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


class JointSineWaveVelocityAction(JointVelocityAction):
    """Joint action term that applies sinusoidal velocity control to the articulation's joints."""
    
    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.amplitude_odd = cfg.amplitude_odd  # 振幅 A
        self.amplitude_even = cfg.amplitude_even  # 振幅 A
        self.frequency_odd = cfg.frequency_odd  # 频率 ω
        self.frequency_even = cfg.frequency_even  # 频率 ω
        self.joint_names = cfg.joint_names
        self.phase_odd = cfg.phase_odd #if isinstance(cfg.phase, list) else [cfg.phase] * len(self._joint_ids)
        # 如果 cfg.phase 是 float，则将其扩展为与关节数目相同的列表
        self.phase_even = cfg.phase_even
        self.sine_offset = cfg.sine_offset  # 偏移 θ0
        self.start_time = time.time()  # 记录初始时间，用于计算当前时间 t
        
    
    def get_joint_index(self, joint_name: str) -> int:
        """根据关节名称返回关节的索引"""
        return self.joint_names.index(joint_name)

    
    def apply_actions(self):
        # 计算当前时间
        current_time = torch.tensor(time.time() - self.start_time, dtype=torch.float32)

        # 创建正弦波速度控制信号
        sine_wave_velocities = torch.zeros(len(self.joint_names), dtype=torch.float32)

        # 对每个关节生成正弦波速度
        for i in range(len(self.joint_names)):
            if i % 2 == 0:  # 偶数关节
                phase_offset = (i / 2) * self.phase_even

                desired_velocity = (
                self.amplitude_even * torch.cos(5 * torch.pi * self.frequency_even * current_time + phase_offset) 
                    + self.sine_offset
                )
                sine_wave_velocities[i] = desired_velocity
            else:  # 奇数关节
                phase_offset = ((i - 1) / 2) * self.phase_odd

                desired_velocity = (
                    self.amplitude_odd * torch.cos(5 * torch.pi * self.frequency_odd * current_time + phase_offset) 
                    + self.sine_offset
                )
                sine_wave_velocities[i] = desired_velocity

        # 设置关节速度目标
        self._asset.set_joint_velocity_target(sine_wave_velocities, joint_ids=self._joint_ids)





# class JointEffortAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

#     cfg: actions_cfg.JointEffortActionCfg
#     """The configuration of the action term."""

#     def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)

#     def apply_actions(self):
#         # set joint effort targets
#         self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)

