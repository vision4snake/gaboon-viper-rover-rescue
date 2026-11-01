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



def reward_cut_point(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command
    radius = 0.35
    curr_xy, targ_xy = current_pos[:, :2], target_pos[:, :2]
    
    # 1. 基础几何计算
    vec_to_center = targ_xy - curr_xy
    dist = torch.norm(vec_to_center, dim=1)
    theta_center = torch.atan2(vec_to_center[:, 1], vec_to_center[:, 0])
    
    # 2. 圆内/外自适应处理 (防崩溃)
    in_circle_mask = dist < radius
    ratio = torch.clamp(radius / (dist + 1e-6), max=1.0)
    theta_offset = torch.where(in_circle_mask, torch.tensor(torch.pi/2).to(dist.device), torch.asin(ratio))
    
    # 3. 自适应判断：哪条切线更顺路
    def wrap_to_pi(a): return (a + torch.pi) % (2 * torch.pi) - torch.pi
    err_ccw = torch.abs(wrap_to_pi(theta_center + theta_offset - robot_heading.squeeze()))
    err_cw = torch.abs(wrap_to_pi(theta_center - theta_offset - robot_heading.squeeze()))
    
    # 4. 计算到切点（圆外）或圆周（圆内）的距离
    # 虽然数学上两个切点距离一样，但逻辑上我们要维持这种自适应选择的一致性
    safe_dist_sq = torch.clamp(dist**2 - radius**2, min=0.0)
    tangent_dist = torch.sqrt(safe_dist_sq)
    final_dist = torch.where(in_circle_mask, torch.abs(dist - radius), tangent_dist)
    
    # 距离越近奖励越高
    return torch.exp(-1.5 * final_dist)



def reward_orbit(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command_new(command_name)
    # command2 = env.command_manager.get_command_data_2(command_name)
    current_pos, target_pos, _, _ = command
    radius = 0.35
    
    # 计算径向距离偏差
    dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=1)
    
    # 高斯奖励：sigma=0.08 确保了 0.25m~0.35m 的精确控制带
    # print("common_step_counter:",env.common_step_counter)

    # print("pos_command_w[0, 0]:",command2[0])
    # print("pos_command_w[0, 1]:",command2[1])
    return torch.exp(-0.5 * ((dist - radius) / 0.08)**2)



def reward_heading(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command_new(command_name)
    current_pos, target_pos, robot_heading, _ = command 
    radius = 0.35
    curr_xy, targ_xy = current_pos[:, :2], target_pos[:, :2]
    
    vec_to_center = targ_xy - curr_xy
    dist = torch.norm(vec_to_center, dim=1)
    theta_center = torch.atan2(vec_to_center[:, 1], vec_to_center[:, 0])
    
    # 自适应计算偏角（圆内强制 90 度以螺旋滑出）
    ratio = torch.clamp(radius / (dist + 1e-6), max=1.0)
    theta_offset = torch.where(dist < radius, torch.tensor(torch.pi/2).to(dist.device), torch.asin(ratio))
    
    # 选取最顺路的切线方向
    def wrap_to_pi(a): return (a + torch.pi) % (2 * torch.pi) - torch.pi
    err_ccw = torch.abs(wrap_to_pi(theta_center + theta_offset - robot_heading.squeeze()))
    err_cw = torch.abs(wrap_to_pi(theta_center - theta_offset - robot_heading.squeeze()))
    
    heading_err = torch.min(err_ccw, err_cw)
    
    # 朝向误差越小，奖励越高
    return torch.exp(-2.0 * heading_err)



# def reward_orthogonal_velocity_phased(env, command_name: str) -> torch.Tensor:
#     """动力学约束：平滑阶段性惩罚径向速度，消除离心侧滑"""
#     command = env.command_manager.get_command_new(command_name)
#     current_pos, target_pos, _, robot_vel_xy_w = command
    
#     radius = 0.35
    
#     # 提取 XY 位置
#     curr_xy = current_pos[:, :2]
#     targ_xy = target_pos[:, :2]
    
#     # 1. 计算径向向量 (均在世界坐标系下)
#     radial_vec_w = curr_xy - targ_xy
#     dist = torch.norm(radial_vec_w, dim=1, keepdim=True)
    
#     # 2. 归一化径向向量
#     unit_radial_vec_w = radial_vec_w / (dist + 1e-6)
    
#     # 3. 计算径向滑动速度 (世界坐标系下的点乘)
#     radial_velocity = torch.sum(robot_vel_xy_w * unit_radial_vec_w, dim=1)
    
#     # 4. 平滑漏斗形激活权重 (sigma=0.08，提供完美的“软着陆”缓冲带)
#     activation_weight = torch.exp(-0.5 * ((dist.squeeze() - radius) / 0.08)**2)
    
#     # 5. 计算最终惩罚
#     sigma_v = 2.0
#     penalty_term = torch.square(radial_velocity)
#     reward = torch.exp(-sigma_v * penalty_term * activation_weight)
    
#     return reward


# def reward_straight_locking(env, command_name: str) -> torch.Tensor:
#     command = env.command_manager.get_command_new(command_name)
#     current_pos, target_pos, robot_heading, _ = command 
#     radius = 0.40 #0.35
#     curr_xy, targ_xy = current_pos[:, :2], target_pos[:, :2]
    
#     dist = torch.norm(targ_xy - curr_xy, dim=1)
#     theta_center = torch.atan2(targ_xy[:, 1] - curr_xy[:, 1], targ_xy[:, 0] - curr_xy[:, 0])
    
#     # 1. 确定当前应该跟随哪条切线
#     ratio = torch.clamp(radius / (dist + 1e-6), max=1.0)
#     theta_offset = torch.where(dist < radius, torch.tensor(torch.pi/2).to(dist.device), torch.asin(ratio))
    
#     def wrap_to_pi(a): return (a + torch.pi) % (2 * torch.pi) - torch.pi
#     err_ccw = torch.abs(wrap_to_pi(theta_center + theta_offset - robot_heading.squeeze()))
#     err_cw = torch.abs(wrap_to_pi(theta_center - theta_offset - robot_heading.squeeze()))
    
#     # 2. 只有当 Heading 已经基本对准目标直线时，才激活“直线锁定”
#     best_heading_err = torch.min(err_ccw, err_cw)
    
#     # 3. 距离权重：离圆心越远（典型 S 段区域），直线保持的奖励越高
#     # 进入圆内后权重降为 0
#     s_weight = torch.clamp(dist - radius, min=0.0, max=1.0)
    
#     # 综合奖励：角度越对齐 + 处于 S 段 = 奖励越高
#     reward = torch.exp(-3.0 * best_heading_err) * s_weight
    
#     return reward

# def reward_straight_locking(env, command_name: str) -> torch.Tensor:
#     command = env.command_manager.get_command_new(command_name)
#     current_pos, target_pos, robot_heading, _ = command 
#     radius = 0.40
#     curr_xy, targ_xy = current_pos[:, :2], target_pos[:, :2]
    
#     dist = torch.norm(targ_xy - curr_xy, dim=1)
#     theta_center = torch.atan2(targ_xy[:, 1] - curr_xy[:, 1], targ_xy[:, 0] - curr_xy[:, 0])
    
#     # 1. 确定当前应该跟随哪条切线
#     ratio = torch.clamp(radius / (dist + 1e-6), max=1.0)
#     theta_offset = torch.where(dist < radius, torch.tensor(torch.pi/2).to(dist.device), torch.asin(ratio))
    
#     def wrap_to_pi(a): return (a + torch.pi) % (2 * torch.pi) - torch.pi
#     err_ccw = torch.abs(wrap_to_pi(theta_center + theta_offset - robot_heading.squeeze()))
#     err_cw = torch.abs(wrap_to_pi(theta_center - theta_offset - robot_heading.squeeze()))
#     best_heading_err = torch.min(err_ccw, err_cw)
    
#     # --- 核心修改部分 ---
#     # 原来是: s_weight = torch.clamp(dist - radius, min=0.0, max=1.0)
#     # 现在的逻辑：只要“对齐了切线”，就给奖。
#     # 我们用一个反向的 sigmoid 或 exp 来衡量“对齐程度”
#     alignment_reward = torch.exp(-3.0 * best_heading_err)
    
#     # 为了让进圈后的曲线不掉下来，我们不再用 (dist - radius) 
#     # 而是奖励“保持在正确轨道上”的状态
#     # 这样即便 dist 减小，只要 alignment_reward 高，总分就高
#     return alignment_reward

# def reward_straight_locking(env, command_name: str) -> torch.Tensor:
#     # 1. 基础数据提取
#     command = env.command_manager.get_command_new(command_name)
#     current_pos, target_pos, robot_heading, _ = command
#     radius = 0.35 #0.40
#     curr_xy, targ_xy = current_pos[:, :2], target_pos[:, :2]
    
#     # 2. 基础几何计算
#     vec_to_center = targ_xy - curr_xy
#     dist = torch.norm(vec_to_center, dim=1)
#     theta_center = torch.atan2(vec_to_center[:, 1], vec_to_center[:, 0])
    
#     # 3. 确定理想切线偏角 (自适应圆内/圆外)
#     # 即使在圆内，也强制 90 度偏角以引导其螺旋向圆周切入
#     ratio = torch.clamp(radius / (dist + 1e-6), max=1.0)
#     theta_offset = torch.where(dist < radius, 
#                                torch.tensor(torch.pi/2).to(dist.device), 
#                                torch.asin(ratio))
    
#     # 4. 计算航向与两条切线中较近者的误差
#     def wrap_to_pi(a): return (a + torch.pi) % (2 * torch.pi) - torch.pi
#     err_ccw = torch.abs(wrap_to_pi(theta_center + theta_offset - robot_heading.squeeze()))
#     err_cw = torch.abs(wrap_to_pi(theta_center - theta_offset - robot_heading.squeeze()))
#     best_heading_err = torch.min(err_ccw, err_cw)
    
#     # 5. 【核心改进】设计平稳的渐变权重 s_weight
#     # 我们不再让它在 radius 处归零，而是让它在 [0.5, 1.0] 之间平滑过渡
#     # 远端 (dist >> radius) -> 趋近 1.0
#     # 近端 (dist <= radius) -> 保持 0.5
    
#     # 使用指数衰减函数创造平滑过渡带 (0.5 为基准增益)
#     # 当 dist-radius 很大时，exp项趋于0，权重趋于1.0
#     # 当 dist-radius <= 0 时，exp项为1，权重为0.5
#     diff = torch.clamp(dist - radius, min=0.0)
#     s_weight = 0.5 + 0.3 * (1.0 - torch.exp(-1.5 * diff))
    
#     # 6. 综合奖励计算
#     # 使用 -3.0 的指数系数确保对直线精度的严格要求
#     reward = torch.exp(-3.0 * best_heading_err) * s_weight
    
#     return reward








