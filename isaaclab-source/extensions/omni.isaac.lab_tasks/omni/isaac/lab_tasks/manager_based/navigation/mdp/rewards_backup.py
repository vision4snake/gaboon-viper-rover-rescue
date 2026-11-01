# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
import numpy as np

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    #print("des_pos_b = command[:, :3]:",command[1, :3])
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh_backup(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    #print("command[1]:",command[1])
    des_quat_b = command[:, 3]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 4]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def reward_cut_point(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, _, _ = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    # print('current_pos[0]:',current_pos[0])
    # print('target_pos[0]:',target_pos[0])
    
    radius = 0.40 #0.40
    
    # 计算从当前点到目标点的方向向量
    direction = current_pos - target_pos
    norm_direction = torch.norm(direction, dim=1, keepdim=True)  # norm_direction 形状为 [4096, 1]
    
    # 将 norm_direction 转换为 [4096] 形状
    norm_direction = norm_direction.squeeze()  # 现在它的形状是 [4096]
    
    # 如果当前点与目标点的距离小于或等于半径，切点与目标点重合，距离切点的距离为 0
    small_distance_mask = norm_direction <= radius  # 如果距离小于等于半径
    
    # 对于距离小于等于半径的情况，距离为 0
    distance_to_cut_point = torch.where(small_distance_mask, torch.zeros_like(norm_direction), torch.full_like(norm_direction, float('inf')))
    
    # 如果距离大于半径，计算切点
    if not torch.all(small_distance_mask):  # 检查是否有任何实例满足距离小于半径的条件
        # 计算单位方向向量
        unit_direction = direction / norm_direction.unsqueeze(1)  # 确保方向的维度是 [4096, 2]
        
        # 计算切线的方向（垂直于单位方向向量）
        tangent_direction = torch.stack([-unit_direction[:, 1], unit_direction[:, 0]], dim=1)
        
        # 计算切点到目标点的距离（根据勾股定理）
        tangent_distance_term = norm_direction**2 - radius**2

        # 检查是否有负数，并打印出来
        # if torch.any(tangent_distance_term < 0):
        #     print("Warning: Negative value for tangent_distance_term detected!")
        #     print("tangent_distance_term:", tangent_distance_term)

        # 计算切点的距离（使用 torch.sqrt）
        tangent_distance = torch.sqrt(torch.maximum(tangent_distance_term, torch.zeros_like(tangent_distance_term)))

        
        # 计算两个切点的位置
        cut_point_1 = target_pos + tangent_direction * tangent_distance.unsqueeze(1)  # 顺时针切点
        cut_point_2 = target_pos - tangent_direction * tangent_distance.unsqueeze(1)  # 逆时针切点
        
        # 计算机器人到两个切点的距离
        distance_to_cut_point_1 = torch.norm(current_pos - cut_point_1, dim=1)
        distance_to_cut_point_2 = torch.norm(current_pos - cut_point_2, dim=1)
        
        # 选择距离最近的切点作为最终的切点
        distance_to_cut_point = torch.min(distance_to_cut_point_1, distance_to_cut_point_2)

    # 奖励：距离切点越近奖励越高
    reward = -distance_to_cut_point  # 距离越近，奖励越高
    # # #reward = 1 - torch.tanh(distance_to_cut_point)
    #reward = 1 - torch.tanh(distance_to_cut_point)
    #reward = -torch.norm(distance_to_cut_point)
    #reward = -torch.tanh(distance_to_cut_point)
    return reward

# def reward_cut_point(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
#     # 从环境中获取命令数据
#     command = env.command_manager.get_command_new(command_name)
#     current_pos, target_pos, _, _ = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    
#     radius = 0.40
    
#     # 计算切点位置 (每个环境的切点)
#     direction = current_pos - target_pos
#     norm_direction = torch.norm(direction, dim=1, keepdim=True)
#     cut_point = target_pos + radius * direction / norm_direction  # (batch_size, 2)
    
#     # 计算机器人到切点的距离
#     distance_to_cut_point = torch.norm(current_pos - cut_point, dim=1)
    
#     # 奖励：距离切点越近奖励越高
#     reward = -distance_to_cut_point  # 距离越近，奖励越高
    
#     return reward

def reward_target(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, _, _ = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    
    # 计算当前位置到圆心的距离
    distance_to_target = torch.norm(current_pos - target_pos, dim=1)
    
    # 如果机器人接近圆心，则给出奖励
    reward = torch.where((distance_to_target >= 0.35) & (distance_to_target < 0.45), torch.tensor(1.0, device=current_pos.device), torch.tensor(0.0, device=current_pos.device))
    
    return reward

def reward_circle(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, _, _ = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    radius = 0.4 #0.4
    
    # 计算机器人当前位置到圆心的距离
    distance_from_center = torch.norm(current_pos - target_pos, dim=1)
    
    # 奖励：距离圆心越接近 1 米奖励越高
    reward = -torch.abs(distance_from_center - radius)  # 离圆心越近，奖励越高
    
    return reward

def reward_heading(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command  # 假设 command 返回的是 (current_pos, target_pos, robot_heading) 格式
    
    # 设置圆的半径（0.4m）
    radius = 0.4
    
    # 计算从当前点到目标点的方向向量
    direction = current_pos - target_pos
    
    # 计算方向向量的长度 (批次操作)
    distance = torch.norm(direction, dim=1, keepdim=True)  # 结果维度是 [4096, 1]
    
    # 如果 distance 小于等于半径时，进行特殊处理
    # 通过比较 distance 是否小于等于半径，生成一个布尔值的掩码
    small_distance_mask = distance <= radius
    
    # 创建最终的奖励变量，初始化为 0
    final_reward = torch.zeros_like(distance).squeeze()  # shape: [4096]
    
    # 如果 distance 小于等于半径时，进行垂直朝向计算
    if torch.any(small_distance_mask):
        # 对于距离小于或等于半径的样本，我们采用垂直方向的朝向
        unit_direction = direction / distance  # 单位方向向量

        # 计算垂直的单位方向（旋转 90 度）
        vertical_direction = torch.stack([-unit_direction[:, 1], unit_direction[:, 0]], dim=1)
        
        # 计算机器人当前朝向的单位向量
        robot_heading_unit = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
        
        # 计算朝向误差（垂直朝向）
        heading_error = torch.abs(torch.sum(vertical_direction * robot_heading_unit, dim=1))  # (batch_size,)
        
        # 对于小于半径的实例，奖励是朝向误差的反向
        # #reward_small = 1.0 - heading_error #torch.tanh(heading_error)  # 朝向误差越小，奖励越高
        # reward_small = 1.0 - torch.tanh(heading_error)
        reward_small = 1.0 - heading_error
        
        # 将小于半径的样本的奖励应用到最终的 reward 向量中
        final_reward = torch.where(small_distance_mask.squeeze(), reward_small, final_reward)
    
    # 对于距离大于半径的样本，继续计算切点
    large_distance_mask = ~small_distance_mask  # 创建大于半径的掩码
    
    if torch.any(large_distance_mask):
        # 计算切线方向和两个切点
        unit_direction = direction / distance
        tangent_direction = torch.stack([-unit_direction[:, 1], unit_direction[:, 0]], dim=1)
        
        # 计算切点到目标点的距离（根据勾股定理）
        tangent_distance_term = distance**2 - radius**2

        # 检查是否有负数，并打印出来
        # if torch.any(tangent_distance_term < 0):
        #     print("Warning: Negative value for tangent_distance_term detected!")
        #     print("tangent_distance_term:", tangent_distance_term)

        # 计算切点的距离（使用 torch.sqrt）
        tangent_distance = torch.sqrt(torch.maximum(tangent_distance_term, torch.zeros_like(tangent_distance_term)))
        
        cut_point_1 = target_pos + tangent_direction * tangent_distance
        cut_point_2 = target_pos - tangent_direction * tangent_distance
        
        distance_to_cut_point_1 = torch.norm(current_pos - cut_point_1, dim=1)
        distance_to_cut_point_2 = torch.norm(current_pos - cut_point_2, dim=1)
        
        # 选择距离最近的切点
        distance_to_cut_point = torch.min(distance_to_cut_point_1, distance_to_cut_point_2)
        
        # 计算朝向误差
        direction_to_cut_point = (cut_point_1 - target_pos) / torch.norm(cut_point_1 - target_pos, dim=1, keepdim=True)
        robot_heading_unit = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
        heading_error = torch.abs(torch.sum(direction_to_cut_point * robot_heading_unit, dim=1))
        
        # #reward_large = 1.0 - heading_error #torch.tanh(heading_error)  # 朝向误差越小，奖励越高
        # reward_large = 1.0 - torch.tanh(heading_error)
        reward_large = 1.0 - heading_error
        
        # 将大于半径的样本的奖励应用到最终的 reward 向量中
        final_reward = torch.where(large_distance_mask.squeeze(), reward_large, final_reward)
    
    return final_reward

def reward_heading_new(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command  # 假设 command 返回的是 (current_pos, target_pos, robot_heading) 格式
    
    # 设置圆的半径（0.4m）
    radius = 0.4 #0.4
    
    # 计算从当前点到目标点的方向向量
    direction = current_pos - target_pos
    
    # 计算方向向量的长度 (批次操作)
    distance = torch.norm(direction, dim=1, keepdim=True)  # 结果维度是 [4096, 1]
    
    # 如果 distance 小于等于半径时，直接给予最大奖励
    small_distance_mask = distance <= radius  # 如果距离小于等于半径
    
    # 创建最终的奖励变量，初始化为 0
    final_reward = torch.zeros_like(distance).squeeze()  # shape: [4096]
    
    # 对于距离小于等于半径的情况，直接给最大奖励
    #final_reward = torch.where(small_distance_mask.squeeze(), torch.zeros_like(final_reward), final_reward)
    final_reward = torch.where(small_distance_mask.squeeze(), torch.ones_like(final_reward), final_reward)

    
    # 对于距离大于半径的情况，继续计算朝向误差
    large_distance_mask = ~small_distance_mask  # 创建大于半径的掩码
    
    if torch.any(large_distance_mask):
        # 计算切线方向和两个切点
        unit_direction = direction / distance
        tangent_direction = torch.stack([-unit_direction[:, 1], unit_direction[:, 0]], dim=1)
        
        # 计算切点到目标点的距离（根据勾股定理）
        tangent_distance_term = distance**2 - radius**2
        
        # 计算切点的距离（使用 torch.sqrt）
        tangent_distance = torch.sqrt(torch.maximum(tangent_distance_term, torch.zeros_like(tangent_distance_term)))
        
        cut_point_1 = target_pos + tangent_direction * tangent_distance
        cut_point_2 = target_pos - tangent_direction * tangent_distance
        
        distance_to_cut_point_1 = torch.norm(current_pos - cut_point_1, dim=1)
        distance_to_cut_point_2 = torch.norm(current_pos - cut_point_2, dim=1)
        
        # 选择距离最近的切点
        #distance_to_cut_point = torch.min(distance_to_cut_point_1, distance_to_cut_point_2)
        
        # 选择最近的切点对应的方向
        # 选择距离最近的切点
        
        # 使用 torch.stack 来将 cut_point_1 和 cut_point_2 放在一起
        cut_points = torch.stack([cut_point_1, cut_point_2], dim=1)  # shape: [4096, 2, 2]
        
        # 使用 torch.argmin 选择距离最近的切点（0 表示第一个切点，1 表示第二个切点）
        chosen_cut_point_idx = (distance_to_cut_point_1 <= distance_to_cut_point_2).long()
        
        # 修正：使用 chosen_cut_point_idx 来选择切点
        chosen_cut_point = torch.gather(cut_points, dim=1, index=chosen_cut_point_idx.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 2))
        
        # 现在 chosen_cut_point 应该是 [4096, 2]，与 target_pos 形状一致
        
        # 计算朝向误差
        direction_to_cut_point = (chosen_cut_point.squeeze(1) - target_pos) / torch.norm(chosen_cut_point.squeeze(1) - target_pos, dim=1, keepdim=True)
        robot_heading_unit = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
        heading_error = torch.abs(torch.sum(direction_to_cut_point * robot_heading_unit, dim=1))
        
        
        # 计算朝向误差的奖励
        reward_large = 1.0 - heading_error
        #reward_large = -torch.tanh(heading_error)
        
        # 将大于半径的样本的奖励应用到最终的 reward 向量中
        final_reward = torch.where(large_distance_mask.squeeze(), reward_large, final_reward)
    
    return final_reward


# def reward_heading(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
#     # 从环境中获取命令数据
#     command = env.command_manager.get_command_new(command_name)
#     current_pos, target_pos, robot_heading, _ = command  # 假设 command 返回的是 (current_pos, target_pos, robot_heading) 格式
    
#     # # 计算切点位置
#     # direction = current_pos - target_pos
#     # norm_direction = torch.norm(direction, dim=1, keepdim=True)
#     # cut_point = target_pos + 1.0 * direction / norm_direction  # (batch_size, 2)
#     radius = 0.4
#     # 计算从当前点到目标点的方向向量
#     direction = current_pos - target_pos
    
#     # 计算方向向量的长度
#     distance = torch.norm(direction, dim=1, keepdim=True)
    
#     # 计算单位方向向量
#     unit_direction = direction / distance
    
#     # 计算切线的方向（旋转 90 度）
#     tangent_direction = torch.stack([-unit_direction[:, 1], unit_direction[:, 0]], dim=1)
    
#     # 计算切点到目标点的距离
#     tangent_distance = torch.sqrt(distance**2 - radius**2)
    
#     # 计算切点的位置
#     cut_point = target_pos + tangent_direction * tangent_distance
    
#     # 计算机器人朝向与理想朝向的误差
#     direction_to_cut_point = (cut_point - target_pos) / torch.norm(cut_point - target_pos, dim=1, keepdim=True)  # (batch_size, 2)
#     robot_heading_unit = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)  # (batch_size, 2)
#     heading_error = torch.abs(torch.sum(direction_to_cut_point * robot_heading_unit, dim=1))  # (batch_size,)
    
#     # 奖励：朝向越一致，奖励越高
#     reward = 1.0 - heading_error  # 朝向误差越小，奖励越高
    
#     return reward

def reward_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, _, _ = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    
    # 计算机器人当前相对于圆心的角度
    angle_covered = torch.atan2(current_pos[:, 1] - target_pos[:, 1], current_pos[:, 0] - target_pos[:, 0])  # (batch_size,)
    
    # 如果机器人已经完成绕一圈，给予奖励
    reward = torch.where(torch.abs(angle_covered) >= 2 * np.pi, torch.tensor(50.0, device=current_pos.device), torch.tensor(0.0, device=current_pos.device))
    
    return reward

def reward_sinwave(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, _, _, joint_vel = command
    # 固定偶数关节的振幅
    amplitude_even = torch.tensor(18.0).to(current_pos.device)
    
    # 固定偶数关节的频率
    frequency_even = 1.4
    
    # 固定偶数关节的相位差
    phase_even = torch.tensor(np.pi / 4).to(current_pos.device)
    
    # 获取当前时间
    #print('env.action_manager._terms:', env.action_manager._terms)
    current_time = env.action_manager.get_term_time('joint_velocity_action')
    #print("current_time:",current_time)
    
    # 计算目标正弦波速度
    sine_wave_velocities = torch.zeros(env.num_envs, 15, device=current_pos.device)
    
    # 计算偶数关节的目标正弦波速度
    for i in range(15):
        if i % 2 == 0:  # 偶数关节
            phase_offset = (i // 2) * phase_even  # 将偶数关节的 phase_offset 计算出来
            # 生成目标正弦波速度
            desired_velocity = amplitude_even * torch.cos(torch.pi * frequency_even * current_time + phase_offset)
            sine_wave_velocities[:, i] = desired_velocity
    
    # 获取实际的关节速度
    actual_velocities = joint_vel  # 这应该是一个 [batch_size, joint_count] 的张量
    
    # 仅取偶数关节的实际速度
    actual_even_velocities = actual_velocities[:, ::2]  # 选择偶数索引的关节速度
    
    # 计算奖励：使用L2范数来计算偶数关节的误差
    error = sine_wave_velocities[:, ::2] - actual_even_velocities  # 计算偶数关节的误差
    reward = torch.norm(error, p=2, dim=1).mean()  # 对所有环境计算误差的L2范数并取平均值，取负数作为奖励

    # 可以加上缩放因子来调整奖励的大小
    scale = 0.1  # 适当的缩放因子
    
    # 计算奖励，L2范数较小，奖励较高
    reward = -scale * reward  # 平均L2范数作为奖励
    
    return reward

def angle_to_target_tangent(current_pos, robot_heading, target_pos, radius):
    """
    计算机器人朝向与目标圆切线方向的角度差。
    - current_pos: 当前机器人位置 [batch_size, 2]
    - robot_heading: 机器人当前朝向 [batch_size]
    - target_pos: 目标圆心位置 [batch_size, 2]
    - radius: 目标圆的半径
    返回:
    - angle_diff: 机器人朝向与目标圆切线方向的角度差 [batch_size]
    """
    # 计算目标圆心到当前点的向量
    to_target = target_pos - current_pos  # [batch_size, 2]
    
    # 计算切线方向（法线旋转 90 度，逆时针）
    tangent_direction = torch.stack([-to_target[:, 1], to_target[:, 0]], dim=1)  # [batch_size, 2]
    tangent_direction = tangent_direction / torch.linalg.norm(tangent_direction, dim=1, keepdim=True)  # 单位化

    # 计算切线方向的角度
    tangent_angle = torch.atan2(tangent_direction[:, 1], tangent_direction[:, 0])  # [batch_size]
    
    # 计算角度差
    angle_diff = tangent_angle - robot_heading  # [batch_size]
    
    # 确保角度差在 [-π, π] 范围内
    angle_diff = torch.where(angle_diff > torch.pi, angle_diff - 2 * torch.pi, angle_diff)
    angle_diff = torch.where(angle_diff < -torch.pi, angle_diff + 2 * torch.pi, angle_diff)
    
    return angle_diff


def orientation_reward_stage1(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    阶段1奖励函数：鼓励蛇形机器人绕初始点的圆弧调整朝向，直到与目标圆的切线方向一致。
    - current_pos: 当前机器人位置 [x, y]
    - robot_heading: 当前机器人朝向
    - target_pos: 目标点位置 [x, y]
    - radius: 初始点所在圆的半径
    - threshold: 朝向的阈值，若朝向接近目标切线，奖励较大
    """
    radius=0.4
    threshold=0.1
     # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, joint_vel = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    
    # 计算机器人朝向与目标切线的角度差
    angle_diff = angle_to_target_tangent(current_pos, robot_heading, target_pos, radius)  # [batch_size]
    
    # 计算奖励：角度差越小奖励越大
    reward = -torch.abs(angle_diff)  # 绝对值越小（越接近对齐），奖励越大
    
    # 归一化奖励（可选）
    #reward_norm = torch.norm(reward, p=2)

    # 保持负号奖励的特性
    #reward = torch.sign(reward) * reward_norm

    return reward

def along_tangent_line_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    阶段2奖励函数：鼓励蛇形机器人沿着公切线行驶到目标圆的切点。
    - current_pos: 当前机器人位置 [x, y]
    - target_pos: 目标点位置 [x, y]
    - radius: 目标圆的半径
    - target_threshold: 目标切点的阈值，接近切点奖励更大
    """
    radius=0.4
    target_threshold=0.1
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, joint_vel = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    # 计算机器人到目标切点的距离
    dist_to_target = torch.linalg.norm(current_pos - target_pos, dim=1)
    
    # 给出奖励，如果离目标较近则奖励较大，否则随着距离增大而减少奖励
    reward = torch.where(dist_to_target < target_threshold,
                         torch.tensor(10.0, device=current_pos.device),  # 机器人接近目标时，奖励较大
                         -dist_to_target)  # 离目标越远，奖励越小
    
    # 归一化处理，避免数值过大
    #reward = torch.norm(reward, p=2)  # 对整个 batch 进行 L2 归一化
    
    return reward

def circular_motion_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    阶段3奖励函数：鼓励蛇形机器人绕目标点的圆弧运动。
    - current_pos: 当前机器人位置 [x, y]
    - target_pos: 目标点位置 [x, y]
    - radius: 目标圆的半径
    - max_distance: 离目标圆半径的最大偏差
    """
    radius=0.4
    max_distance=0.1
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, joint_vel = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    # 计算机器人到目标圆心的距离
    dist_to_target = torch.linalg.norm(current_pos - target_pos, dim=1)  # 计算每个机器人到目标圆心的距离
    
    # 计算当前点与目标圆的半径的偏差
    dist_to_circle = torch.abs(dist_to_target - radius)  # 偏差是到目标圆心距离减去半径
    
    # 根据偏差计算奖励
    reward = torch.where(
        dist_to_circle < max_distance,
        torch.tensor(1.0, device=current_pos.device),  # 偏差小于允许值，给予奖励
        -dist_to_circle  # 偏差越大，奖励越小
    )
    
    # 归一化处理（可选，根据具体需求）
    #reward = torch.norm(reward, p=2)

    return reward

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2_odd(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the odd-indexed joints using L2 squared kernel.

    NOTE: Only the odd-indexed joints will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Filter for odd-indexed joints
    odd_joint_ids = [joint_id for joint_id in asset_cfg.joint_ids if joint_id % 2 == 1]
    
    # Calculate the L2 penalty for the odd-indexed joint accelerations
    return torch.sum(torch.square(asset.data.joint_acc[:, odd_joint_ids]), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_torques_l2_odd(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the odd-indexed joints using L2 squared kernel.

    NOTE: Only the odd-indexed joints will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Filter for odd-indexed joints
    odd_joint_ids = [joint_id for joint_id in asset_cfg.joint_ids if joint_id % 2 == 1]
    
    # Calculate the L2 penalty for the odd-indexed joint torques
    return torch.sum(torch.square(asset.data.applied_torque[:, odd_joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

def joint_vel_l2_odd(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the odd-indexed joints using L2 squared kernel.

    NOTE: Only the odd-indexed joints will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Filter for odd-indexed joints
    odd_joint_ids = [joint_id for joint_id in asset_cfg.joint_ids if joint_id % 2 == 1]
    
    # Calculate the L2 penalty for the odd-indexed joint velocities
    return torch.sum(torch.square(asset.data.joint_vel[:, odd_joint_ids]), dim=1)

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

def reward_target_2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, _, _ = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    
    # 计算当前位置到圆心的距离
    distance_to_target = torch.norm(current_pos - target_pos, dim=1)
    
    return distance_to_target

# 计算切线角度（批量计算）
def calculate_tangent_angles(current_pos, target_pos, r):
    # 计算每个点到圆心的距离 D
    D = torch.norm(current_pos - target_pos, dim=1)

    # 如果 D 小于半径 r，则无法存在切线
    valid = D >= r

    # 计算切线的角度偏移（批量计算）
    angle_to_center = torch.atan2(current_pos[:, 1] - target_pos[:, 1], current_pos[:, 0] - target_pos[:, 0])
    angle_offset = torch.asin(r / D)

    # 计算两条切线的角度
    angle1 = angle_to_center + angle_offset
    angle2 = angle_to_center - angle_offset

    return valid, angle1, angle2

# 判断初始点朝向与切线的夹角（批量计算）
def choose_best_tangent(h, angle1, angle2):
    # 计算每条切线的夹角
    angle_diff1 = torch.abs(h - angle1)
    angle_diff2 = torch.abs(h - angle2)

    # 选择夹角较小的切线
    best_angle = torch.where(angle_diff1 < angle_diff2, angle1, angle2)
    return best_angle

# 计算点到直线的距离
def distance_to_line(current_pos, best_angle):
    # 计算切线的参数 a, b, c
    a = torch.cos(best_angle)
    b = torch.sin(best_angle)
    c = -a * current_pos[:, 0] - b * current_pos[:, 1]

    # 计算每个点到切线的距离
    numerator = torch.abs(a * current_pos[:, 0] + b * current_pos[:, 1] + c)
    denominator = torch.sqrt(a**2 + b**2)

    distance = numerator / denominator
    return distance

def reward_trajectory(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, heading_w, current_time = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    r = 0.4  # 圆的半径

    # 初始化距离
    distances = torch.zeros_like(current_time)
    print("current_time[0]:",current_time[0])
    # 判断每个时间步是否为0
    is_first_time = current_time == 0

    if is_first_time.any():  # 对于时间步长为0的样本，计算切线和初始奖励
        # 批量计算切线角度
        valid, angle1, angle2 = calculate_tangent_angles(current_pos[is_first_time], target_pos[is_first_time], r)

        # 选择与初始朝向最接近的切线
        best_angle = choose_best_tangent(heading_w[is_first_time], angle1, angle2)

        # 当前时间步的距离默认是0，因为是初始状态
        distances[is_first_time] = torch.zeros_like(heading_w[is_first_time])  # 你也可以根据需要设定为其他值
    else:
        # 对于时间步长不为0的样本，计算它们的位置与目标圆心的距离
        distance_from_center = torch.norm(current_pos - target_pos, dim=1)

        # 宽松的容忍度 epsilon = 0.05
        epsilon = 0.05
        in_circle = torch.abs(distance_from_center - r) < epsilon
        print("in_circle.shape:",in_circle.shape)
        print("distance_from_center.shape:",distance_from_center.shape)
        print("distances:",distances)

        # 如果当前点在圆形轨迹上，用圆形轨迹的损失
        distances[in_circle] = torch.abs(distance_from_center[in_circle] - r)  # 距离圆心越近，奖励越高

        # 对于不在圆轨迹上的点，计算它们与切线的距离
        distances[~in_circle] = distance_to_line(current_pos[~in_circle], best_angle[~in_circle])

        max_distance=1.0

    return torch.exp(-distance / max_distance)

# 计算当前点与切点的距离
def distance_to_cut_point(current_pos, cut_point):
    return torch.norm(current_pos - cut_point, dim=1)

# 奖励函数：鼓励机器人靠近切点
def reward_closer_to_cut_point(current_pos, cut_point, max_distance=1.0):
    # 计算当前点到切点的距离
    distance = distance_to_cut_point(current_pos, cut_point)
    
    # 设定奖励的计算方式，距离越近奖励越高
    rewards = torch.exp(-distance / max_distance)  # 使用指数衰减，奖励随距离减少而减小
    
    return rewards

# 判断是否在圆轨迹上
def is_on_circle_trajectory(current_pos, target_pos, r, epsilon=0.05):
    # 计算当前点与圆心的距离
    distance_from_center = torch.norm(current_pos - target_pos, dim=1)
    
    # 判断当前点是否在圆轨迹上（宽松的容忍度 epsilon）
    return torch.abs(distance_from_center - r) < epsilon

def calculate_tangent_intersection(current_pos, target_pos, best_angle, r):
    """
    计算从初始点到目标圆的切点（交点）
    
    参数:
    - current_pos: 初始点的坐标 (B, 2) tensor
    - target_pos: 圆心的位置 (B, 2) tensor
    - best_angle: 选择的切线角度 (B, ) tensor
    - r: 圆的半径 scalar

    返回:
    - cut_point: 切点的坐标 (B, 2) tensor
    """
    # 计算切线的方向
    dx = torch.cos(best_angle)
    dy = torch.sin(best_angle)

    # 计算当前点与目标圆心的距离 D
    D = torch.norm(current_pos - target_pos, dim=1)

    # 计算切点的参数 t（D 是从当前点到目标圆心的距离）
    t = (r - D) / (dx**2 + dy**2)

    # 计算切点的坐标
    cut_point_x = current_pos[:, 0] + t * dx
    cut_point_y = current_pos[:, 1] + t * dy

    # 返回切点的坐标 (B, 2)
    cut_point = torch.stack([cut_point_x, cut_point_y], dim=1)
    
    return cut_point

# 主奖励函数
def reward_point(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, heading_w, current_time = command  # 假设 command 返回的是 (current_pos, target_pos, _) 格式
    r = 0.4  # 圆的半径

    # 计算切点
    valid, angle1, angle2 = calculate_tangent_angles(current_pos, target_pos, r)
    best_angle = choose_best_tangent(heading_w, angle1, angle2)
    cut_point = calculate_tangent_intersection(current_pos, target_pos, best_angle, r)

    # 判断是否在圆轨迹上
    on_circle = is_on_circle_trajectory(current_pos, target_pos, r)

    # 计算奖励
    rewards = torch.zeros_like(current_pos[:, 0])  # 初始化奖励为0

    # 如果不在圆轨迹上，鼓励靠近切点
    if not on_circle.any():  
        rewards = reward_closer_to_cut_point(current_pos, cut_point)  # 鼓励靠近切点

    return rewards



# 计算目标点与当前位置的距离（支持批量计算，使用 PyTorch 张量）
def distance_to_target(current_pos, target_pos):
    return torch.norm(current_pos - target_pos, dim=1)  # dim=1 计算每一行的距离

# 计算当前位置与目标螺旋轨迹点的偏差（支持批量计算，使用 PyTorch 张量）
def distance_to_trajectory(current_pos, progress, r0, target_pos):
    # 计算理想的螺旋轨迹点
    theta = 4 * torch.pi * progress  # 进度基于角度
    r = r0 - (r0 - 0.3) * progress  # 半径根据进度变化 0.2
    # 计算目标螺旋轨迹点的位置
    # print('target_pos[:, 0].unsqueeze(1):',target_pos[:, 0].unsqueeze(1).shape)
    # print('r:',r.shape)
    # print('theta:',theta.shape)
    x = (target_pos[:, 0] + r * torch.cos(theta)).unsqueeze(1)
    y = (target_pos[:, 1] + r * torch.sin(theta)).unsqueeze(1)
    trajectory_points = torch.cat([x, y], dim=1)
    # print("x:",x.shape)
    # print("y:",y.shape)
    # print("r0:",r0.shape)
    # print("trajectory_points:",trajectory_points.shape)
    # 计算当前位置与目标轨迹点的距离
    return torch.norm(current_pos - trajectory_points, dim=1)

# 计算当前位置的半径与目标半径的偏差（支持批量计算，使用 PyTorch 张量）
# def radius_deviation(current_pos, target_pos, r0):
#     # 计算当前位置的半径
#     r_current = torch.sqrt((current_pos[:, 0] - target_pos[:, 0]) ** 2 + (current_pos[:, 1] - target_pos[:, 1]) ** 2)
#     return torch.abs(r_current - 0.3)  # 与目标半径（0.2）之间的偏差
def radius_deviation(current_pos, target_pos, r0, progress):
    r = r0 - (r0 - 0.3) * progress  # 半径随进度变化
    r_current = torch.sqrt((current_pos[:, 0] - target_pos[:, 0]) ** 2 + (current_pos[:, 1] - target_pos[:, 1]) ** 2)
    return torch.abs(r_current - r)


# 计算进度，基于当前位置与目标点的距离（支持批量计算，使用 PyTorch 张量）
def calculate_progress(current_pos, target_pos, r0):
    # 当前位置与目标点的距离
    D = distance_to_target(current_pos, target_pos)
    # 进度计算：距离越小，进度越接近1
    max_distance = r0  # 初始最大距离，等于起始半径
    progress = 1 - (D / max_distance)
    #print('progress:',progress.shape).unsqueeze(1)
    return torch.clamp(progress, 0, 1)  # 确保进度在0到1之间

def reward_spiral(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command  # 假设 command 返回的是 (current_pos, target_pos, robot_heading) 格式
    e = current_pos.shape[0]
    ref_point = torch.Tensor([-0.2, -0.4]).to(current_pos.device)
    ref_point_expanded = ref_point.unsqueeze(0).expand(e, -1)
    r0 = torch.norm(target_pos - ref_point_expanded, dim=1)
    alpha=1
    beta=1
    gamma=1
    # 计算进度
    progress = calculate_progress(current_pos, target_pos, r0)
    
    # 计算各项偏差
    D_distance = distance_to_target(current_pos, target_pos)
    D_trajectory = distance_to_trajectory(current_pos, progress, r0, target_pos)
    D_radius = radius_deviation(current_pos, target_pos, r0, progress)
    
    # 计算奖励
    reward = - torch.tanh(alpha * D_distance + beta * D_trajectory + gamma * D_radius)
    #print('reward:',reward.shape)
    return reward

def reward_spiral_tracking(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    螺旋线跟踪奖励函数：只有当蛇形机器人距离石头小于 0.45 时，才考虑螺旋跟踪奖励。
    奖励值根据距离石头的距离、螺旋半径误差和朝向误差进行计算。
    使用 L2 正则化确保奖励值不会过大。
    """
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command  # shape: [N, 2], [N, 2], [N], [_]
    
    # 石头位置及螺旋参数
    r_start = torch.tensor(0.4)  # 起始半径
    r_end = torch.tensor(0.3)    # 目标半径
    stone_center = target_pos  # 螺旋中心
    alpha = torch.log(r_start / r_end) / (2 * torch.pi)  # 控制螺旋收缩速率
    
    # 计算当前位置相对于石头的极坐标
    rel_pos = current_pos - stone_center
    r_actual = torch.norm(rel_pos, dim=1)  # 当前半径
    theta_actual = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])  # 当前角度
    
    # 计算理论螺旋半径（根据角度变化，按对数螺旋计算）
    r_expected = r_start * torch.exp(-alpha * (theta_actual - theta_actual[0]))
    
    # --- (1) 距离判断 ---
    distance_to_stone = torch.norm(current_pos - stone_center, dim=1)
    within_range_mask = distance_to_stone < 0.42  # 距离小于 0.45 时才计算奖励

    # 计算奖励的初始值
    reward = torch.zeros_like(distance_to_stone)  # 默认为 0
    
    # --- (2) 计算半径误差 ---
    radius_error = torch.abs(r_actual - r_expected)
    radius_reward = torch.exp(-5 * radius_error)  # 半径偏差越小奖励越大
    
    # --- (3) 计算朝向误差 ---
    # 计算螺旋切线方向：tangent = (-sinθ, cosθ)
    tangent_dir = torch.stack([-torch.sin(theta_actual), torch.cos(theta_actual)], dim=1)
    robot_heading_unit = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
    heading_error = 1 - torch.abs(torch.sum(tangent_dir * robot_heading_unit, dim=1))
    heading_reward = torch.exp(-5 * heading_error)
    
    # --- (4) 综合奖励 ---
    reward_within_range = 0.6 * radius_reward + 0.4 * heading_reward  # 在范围内时的奖励
    
    # 只有当距离小于 0.45 时，才考虑螺旋跟踪奖励
    reward = torch.where(within_range_mask, reward_within_range, reward)

    # reward = reward.float()
    
    # # --- (5) L2 正则化（通过奖励的 L2 范数来进行正则化）---
    # # 计算奖励的 L2 范数（也就是奖励的平方和）
    # reward_norm = torch.norm(reward, p=2)  # L2 范数
    # regularized_reward = reward / (reward_norm + 1e-8)  # 防止除以零的情况
    
    return reward #regularized_reward

def reward_spiral_tracking_2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    螺旋线跟踪奖励函数：蛇形机器人与目标位置的距离越小，奖励越大；
    且朝向与目标连线垂直时，奖励更高；只有在机器人距离目标小于 0.42 时才计算奖励。
    """
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command  # shape: [N, 2], [N, 2], [N], [_]
    
    # 石头位置（目标位置）
    stone_center = target_pos  # 螺旋中心（目标位置）
    
    # 计算机器人到目标位置的距离
    distance_to_target = torch.norm(current_pos - stone_center, dim=1)
    
    # --- (1) 距离判断 ---
    within_range_mask = distance_to_target <=  0.40  # 距离小于 0.42 时才计算奖励
    reward = torch.zeros_like(distance_to_target)  # 默认为 0，当不在范围内时奖励为 0
    
    # 只有当距离小于 0.42 时，才考虑计算奖励
    if within_range_mask.any():
        # --- (2) 距离接近奖励 ---
        # 距离越接近目标位置，奖励越高
        #distance_reward = 1 - torch.tanh(distance_to_target)  # 距离越小，奖励越大
        distance_reward = 1 - torch.tanh(torch.abs(distance_to_target - 0.3))
        
        # --- (3) 朝向误差奖励 ---
        # 计算当前位置与目标位置连线的方向向量
        rel_pos = current_pos - stone_center
        target_direction = torch.stack([rel_pos[:, 0], rel_pos[:, 1]], dim=1)  # 目标连线方向向量
        
        # 计算机器人朝向的单位向量
        robot_heading_unit = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
        
        # 计算目标方向与机器人朝向之间的夹角
        dot_product = torch.sum(target_direction * robot_heading_unit, dim=1)
        perpendicular_error = torch.abs(dot_product)  # 计算目标方向与机器人朝向的夹角（越小越垂直）
        perpendicular_reward = 1 - torch.tanh(perpendicular_error)  # 越接近垂直，奖励越高
        
        # --- (4) 综合奖励 ---
        reward_within_range = 0.7 * distance_reward + 0.3 * perpendicular_reward  # 根据距离和朝向的权重计算综合奖励
        
        # 计算最终奖励，仅当在距离范围内时才应用
        reward = torch.where(within_range_mask, reward_within_range, reward)

    return reward


def reward_spiral_tracking_3(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    螺旋线跟踪奖励函数（并行环境版本）：
    - 蛇形机器人与目标位置的距离逐渐减小，朝向与目标连线垂直时奖励更高；
    - 只有在机器人距离目标小于 0.42 时才计算奖励；
    - 随着时间推移，每个环境的目标半径从 0.4 逐渐减小到 0.3；
    - 自动记录每个环境第一次进入目标范围的时间 t0。
    """
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command  # shape: [N, 2], [N, 2], [N], [_]
    
    # 目标位置（石头位置）
    stone_center = target_pos  # 螺旋中心（目标位置）
    
    # 计算机器人到目标位置的距离（即半径）
    distance_to_target = torch.norm(current_pos - stone_center, dim=1)  # shape: [N]
    
    # 当前时间步 t，shape: [N]
    t = env.episode_length_buf * env.step_dt
    
    # # --- 记录每个环境第一次进入目标范围的 t0 ---
    # new_t0_mask = torch.isnan(env.t0) & (distance_to_target <= 0.4)
    # env.t0[new_t0_mask] = t[new_t0_mask]
    
    # 设定目标半径变化的开始时间 t0
    t0 = 20  # 例如 t0 设置为 20，可以根据需要调整
    
    # 目标半径随着时间变化从 0.4 减小到 0.3，t >= t0 时开始变化
    target_radius = torch.where(
        t >= t0, 
        0.4 - (0.4 - 0.2) * ((t - t0) / (100 - t0)),  # 目标半径随时间变化
        0.4  # 在 t < t0 时，目标半径保持 0.4
    )
    
    # --- 距离判断 --- 只有当距离小于 0.4 时才计算奖励
    within_range_mask = distance_to_target <= 0.4
    reward = torch.zeros_like(distance_to_target)
    
    if within_range_mask.any():
        # --- 距离奖励 ---
        distance_reward = 1 - torch.tanh(torch.abs(distance_to_target - target_radius))
        
        # --- 朝向奖励 ---
        # rel_pos = current_pos - stone_center
        # target_direction = torch.stack([rel_pos[:, 0], rel_pos[:, 1]], dim=1)
        # robot_heading_unit = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
        # dot_product = torch.sum(target_direction * robot_heading_unit, dim=1)
        # perpendicular_error = torch.abs(dot_product)
        # perpendicular_reward = 1 - torch.tanh(perpendicular_error)
        
        # --- 综合奖励 ---
        reward_within_range = distance_reward #0.6 * distance_reward + 0.4 * perpendicular_reward
        
        # 仅在距离范围内应用奖励
        reward = torch.where(within_range_mask, reward_within_range, reward)
    
    return reward

def reward_spiral_tracking_4(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    螺旋线跟踪奖励函数（并行环境版本）：
    - 蛇形机器人与目标位置的距离逐渐减小，朝向与目标连线垂直时奖励更高；
    - 只有在机器人距离目标小于 0.42 时才计算奖励；
    - 随着时间推移，每个环境的目标半径从 0.4 逐渐减小到 0.3；
    - 自动记录每个环境第一次进入目标范围的时间 t0。
    """
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command  # shape: [N, 2], [N, 2], [N], [_]
    
    # 目标位置（石头位置）
    stone_center = target_pos  # 螺旋中心（目标位置）
    
    # 计算机器人到目标位置的距离（即半径）
    distance_to_target = torch.norm(current_pos - stone_center, dim=1)  # shape: [N]
    
    
    # --- 距离判断 --- 只有当距离小于 0.4 时才计算奖励
    within_range_mask = distance_to_target <= 0.4
    reward = torch.zeros_like(distance_to_target)
    
    if within_range_mask.any():
        # --- 距离奖励 ---
        
        
        # --- 朝向奖励 ---
        rel_pos = current_pos - stone_center
        target_direction = torch.stack([rel_pos[:, 0], rel_pos[:, 1]], dim=1)
        robot_heading_unit = torch.stack([torch.cos(robot_heading), torch.sin(robot_heading)], dim=1)
        dot_product = torch.sum(target_direction * robot_heading_unit, dim=1)
        perpendicular_error = torch.abs(dot_product)
        perpendicular_reward = 1 - torch.tanh(perpendicular_error)
        
        # --- 综合奖励 ---
        reward_within_range = perpendicular_reward
        
        # 仅在距离范围内应用奖励
        reward = torch.where(within_range_mask, reward_within_range, reward)
    
    return reward



def reward_spiral_completion(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    判断蛇形机器人是否完成目标：
    - 机器人距离石头的位置在 0.28 到 0.32 范围内；
    - 机器人处于螺旋线附近，即距离石头的距离在螺旋线的有效范围内。
    """
    # 从环境中获取命令数据
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, _, _ = command  # 假设 command 返回的是 (current_pos, target_pos, robot_heading) 格式

    # 计算蛇形机器人到石头位置的距离
    distance_to_stone = torch.norm(current_pos - target_pos, dim=1)
    
    # 螺旋线的起始半径和结束半径
    r_start = torch.tensor(0.4)  # 起始半径
    r_end = torch.tensor(0.3)    # 目标半径
    epsilon = 0.02  # 螺旋线附近的容差范围

    # 计算当前角度（假设当前角度已在命令中提供）
    angle = torch.atan2(current_pos[:, 1] - target_pos[:, 1], current_pos[:, 0] - target_pos[:, 0])

    # 计算螺旋线的当前半径（根据角度变化）
    alpha = torch.log(r_start / r_end) / (2 * torch.pi)  # 螺旋线的收缩速率
    r_expected = r_start * torch.exp(-alpha * angle)  # 螺旋线的当前半径 

    # 判断是否在螺旋线附近（即距离螺旋线的半径范围内）
    is_near_spiral = (distance_to_stone >= (r_expected - epsilon)) & (distance_to_stone <= (r_expected + epsilon))

    # 判断目标是否达成：距离石头的距离在 0.28 到 0.32 范围内，且在螺旋线附近
    goal_reached = (distance_to_stone >= 0.28) & (distance_to_stone <= 0.32) & is_near_spiral
    
    # 如果目标达成，给予奖励 1，否则奖励 0
    reward = torch.where(goal_reached, torch.ones_like(goal_reached), torch.zeros_like(goal_reached))

    # reward = reward.float()

    # # --- L2 正则化 ---
    # reward_norm = torch.norm(reward, p=2)  # 计算奖励的 L2 范数
    # regularized_reward = reward / (reward_norm + 1e-8)  # 防止除零错误，做正则化处理

    return reward #regularized_reward












