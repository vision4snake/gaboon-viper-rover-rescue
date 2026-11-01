# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
import math
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg, UniformPose2dPRCommandCfg


class UniformPose2dCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: UniformPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.rover_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        #return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)
        #return self.pos_command_b
        return self.pos_command_w - self._env.scene.env_origins

    @property
    def command_data(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        #return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)
        #return self.pos_command_b
        return self.pos_command_w[0] - self._env.scene.env_origins[0]
    
    @property
    def command_new(self): #-> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2], self.pos_command_w[:, :2] - self._env.scene.env_origins[:, :2], self.robot.data.heading_w, self.robot.data.root_lin_vel_w[:, :2]#self.robot.data.joint_vel
        #return self.robot.data.root_pos_w[:, :2], self.pos_command_w[:, :2], self.robot.data.heading_w, self._env.scene.episode_length_buf
        #return self.robot.data.root_pos_w[:, :2], self.pos_command_w[:, :2], self.robot.data.heading_w, self.robot.data.joint_vel
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        #print('self.robot.data:',self.robot.data)
        # 获取所有 body 的名称
        #body_names = self.robot.data.body_names

        # 获取所有 body 的四元数（即旋转信息）
        #body_quaternions = self.robot.data.body_pos_w  # 形状是 (num_instances, num_bodies, 4)

        # 假设我们有一个实例，通常只会有一个实例（num_instances=1），所以索引为0
        # 遍历所有的 body 并打印它们的名称和四元数
        
            # 获取每个 body 对应的四元数
        #quat = body_quaternions[0, 0]  # 选择第一个实例（索引 0）
        #quat1 = body_quaternions[0, 1]  # 选择第一个实例（索引 0）
        #print(f"Body name: {body_names[0]}, Position: {quat}")
        #print(f"Body name: {body_names[1]}, Position: {quat1}")
        
        #print("self.robot.data.body_quat_w:",self.robot.data.root_pos_w[0])
        head_pos = self.robot.data.body_pos_w[:,1]

        #self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        
        #self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
        self.metrics["error_heading"] = self.heading_command_b[:]

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        self.rover_pos[env_ids] = self._env.scene.env_origins[env_ids]
        #print("self.pos_command_w[env_ids]:",self.pos_command_w[env_ids])
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)
        # 随机选择一个区间
        #x_range = random.choice(self.cfg.ranges.pos_x)  # 从两个区间中随机选择一个

        # 使用选中的区间生成随机值
        #self.pos_command_w[env_ids, 0] += r.uniform_(*x_range)  # 解包区间并传给uniform_

        # 随机选择一个区间
        #y_range = random.choice(self.cfg.ranges.pos_y)  # 从两个区间中随机选择一个

        # 使用选中的区间生成随机值
        #self.pos_command_w[env_ids, 1] += r.uniform_(*y_range)  # 解包区间并传给uniform_


        # curriculum_threshold = 500 * 8

        # # success
        # if self.env.common_step_counter < curriculum_threshold:  # 第一阶段：极近距离
        #     pos_x_range = (-0.6, -0.5)
        #     pos_y_range = (-0.45, -0.35)
        # else:                   # 第二阶段：目标距离
        #     pos_x_range = (-0.9, -0.7)
        #     pos_y_range = (-0.7, -0.1)

        # # if self.env.common_step_counter < curriculum_threshold:  # 第一阶段：新手村 (距离约 0.4m)
        # #     # 目标中心选在 x = -0.6 (即 -0.2 + (-0.4))
        # #     # 这样机器人重置时，距离圆心刚好是 0.4m，直接进入 Orbit 状态
        # #     pos_x_range = (-0.65, -0.55) 
        # #     pos_y_range = (-0.45, -0.35) 
        # # else:                                                   # 第二阶段：全地图 (距离约 0.9m)
        # #     # 这是你要求的最终范围，中心距离机器人大约 0.9m
        # #     # 机器人需要爬行约 0.5m (0.9 - 0.4) 才能接触到圆周
        # #     pos_x_range = (-1.2, -1.0)
        # #     pos_y_range = (-0.9, 0.1)


        # self.pos_command_w[env_ids, 0] += r.uniform_(*pos_x_range)
        # self.pos_command_w[env_ids, 1] += r.uniform_(*pos_y_range)

        # print("common_step_counter:",self.env.common_step_counter)

        # print("pos_command_w[0, 0]:",r.uniform_(*pos_x_range)[0])
        # print("pos_command_w[0, 1]:",r.uniform_(*pos_y_range)[0])

        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_w[env_ids, 2] += 0.1 #self.robot.data.default_root_state[env_ids, 2]
        self.rover_pos[env_ids, 0] += 1.2 #0.0 
        self.rover_pos[env_ids, 1] += 0.2 #-1.2
        self.rover_pos[env_ids, 2] = 0.24 #-100#26-6-3 0.24 #seq1 0.22  #0 #0.15

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        #head_pos = self.robot.data.body_pos_w[:,1]
        #target_vec = self.pos_command_w - head_pos[:, :3]
        #self.pos_command_b[:] = quat_rotate_inverse(f(self.robot.data.root_quat_w), target_vec)
        #head_quat = self.robot.data.body_quat_w[:,1]
        #self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(head_quat), target_vec)
        #self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        # 1. 获取头部的位置（只取x, y坐标）
        head_pos = self.robot.data.body_pos_w[:, 0]  # 只获取头部位置的x, y分量
        
        # 2. 计算目标点与头部之间的位置差（只取x, y坐标）
        target_vec = self.pos_command_w - head_pos[:, :3]  # 目标点的位置 - 头部位置
        
        # 3. 获取头部的朝向四元数
        head_quat = self.robot.data.body_quat_w[:, 0]
        
        # 获取头部朝向的前进向量（在XY平面内）
        head_forward_vector = math_utils.quat_apply(head_quat, self.robot.data.FORWARD_VEC_B)
        #head_forward_vector_2d = head_forward_vector[:, :2]  # 只保留x, y分量
        
        # 4. 计算目标点方向与头部朝向之间的夹角（仅在2D平面中）
        dot_product = torch.sum(target_vec * head_forward_vector, dim=1)  # 计算点积
        target_vec_norm = torch.norm(target_vec, dim=1)  # 目标向量的模
        head_forward_vector_norm = torch.norm(head_forward_vector, dim=1)  # 头部前进向量的模
        
        # 计算夹角的余弦值，避免数值误差（限制在[-1, 1]）
        cos_angle = torch.clamp(dot_product / (target_vec_norm * head_forward_vector_norm), -1.0, 1.0)
        
        # 计算夹角（单位：弧度）
        angle = torch.acos(cos_angle)
        
        # 5. 计算目标点相对于头部的方向指令
        # 将目标位置转换到头部的坐标系下（只保留x, y坐标）
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(head_quat), target_vec)  # 只保留x, y分量

        # 6. 计算头部的朝向指令（相对于目标点）
        self.heading_command_b[:] = wrap_to_pi(angle)  # 夹角转换为[-pi, pi]范围内


    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # # update the box marker
        # self.goal_pose_visualizer.visualize(
        #     translations=torch.cat((self.pos_command_w, self.rover_pos), dim=0),
        #     orientations=quat_from_euler_xyz(
        #         torch.cat((torch.zeros_like(self.heading_command_w), torch.zeros_like(self.heading_command_w)), dim=0),
        #         torch.cat((torch.zeros_like(self.heading_command_w), torch.zeros_like(self.heading_command_w)), dim=0),
        #         torch.cat((self.heading_command_w, torch.zeros_like(self.heading_command_w)), dim=0),
        #     ),
        #     marker_indices = [0] * self.num_envs + [1] * self.num_envs
        # )


        # 1. 定义固定的火星车朝向 (你可以随时在这里修改这个值)
        fixed_yaw = 4 * math.pi / 7  #seq1 4 * math.pi / 6 

        # 2. 生成固定朝向张量 (保持和 heading_command_w 同 shape 和同 device)
        fixed_rover_heading = torch.full_like(self.heading_command_w, fixed_yaw)

        # 3. 更新可视化标记 (update the box marker)
        self.goal_pose_visualizer.visualize(
            translations=torch.cat((self.pos_command_w, self.rover_pos), dim=0),
            orientations=quat_from_euler_xyz(
                torch.cat((torch.zeros_like(self.heading_command_w), torch.zeros_like(self.heading_command_w)), dim=0),
                torch.cat((torch.zeros_like(self.heading_command_w), torch.zeros_like(self.heading_command_w)), dim=0),
                # 在这里用 fixed_rover_heading 替换掉第二个 self.heading_command_w
                torch.cat((self.heading_command_w, fixed_rover_heading), dim=0),
            ),
            marker_indices = [0] * self.num_envs + [1] * self.num_envs
        )

        #self.goal_pose_visualizer.visualize(
         #   translations=self.pos_command_w,
          #  orientations=quat_from_euler_xyz(
           #     torch.zeros_like(self.heading_command_w),
            #    torch.zeros_like(self.heading_command_w),
           #     self.heading_command_w,
           # ),
        #)


class TerrainBasedPose2dCommand(UniformPose2dCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]
        # offset the position command by the current root height
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

class UniformPose2dPRCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: UniformPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.rover_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        #return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)
        #return self.pos_command_b
        return self.pos_command_w - self._env.scene.env_origins

    @property
    def command_data(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        #return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)
        #return self.pos_command_b
        return self.pos_command_w[0] - self._env.scene.env_origins[0]
    
    @property
    def command_data_2(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        #return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)
        #return self.pos_command_b
        return self.pos_command_w[0, 0], self.pos_command_w[0, 1]
    
    @property
    def command_new(self): #-> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2], self.pos_command_w[:, :2] - self._env.scene.env_origins[:, :2], self.robot.data.heading_w, self.robot.data.joint_vel
        #return self.robot.data.root_pos_w[:, :2], self.pos_command_w[:, :2], self.robot.data.heading_w, self._env.scene.episode_length_buf

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        #print('self.robot.data:',self.robot.data)
        # 获取所有 body 的名称
        #body_names = self.robot.data.body_names

        # 获取所有 body 的四元数（即旋转信息）
        #body_quaternions = self.robot.data.body_pos_w  # 形状是 (num_instances, num_bodies, 4)

        # 假设我们有一个实例，通常只会有一个实例（num_instances=1），所以索引为0
        # 遍历所有的 body 并打印它们的名称和四元数
        
            # 获取每个 body 对应的四元数
        #quat = body_quaternions[0, 0]  # 选择第一个实例（索引 0）
        #quat1 = body_quaternions[0, 1]  # 选择第一个实例（索引 0）
        #print(f"Body name: {body_names[0]}, Position: {quat}")
        #print(f"Body name: {body_names[1]}, Position: {quat1}")
        
        #print("self.robot.data.body_quat_w:",self.robot.data.root_pos_w[0])
        head_pos = self.robot.data.body_pos_w[:,1]

        #self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        
        #self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
        self.metrics["error_heading"] = self.heading_command_b[:]

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        self.rover_pos[env_ids] = self._env.scene.env_origins[env_ids]
        #print("self.pos_command_w[env_ids]:",self.pos_command_w[env_ids])
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)
        # 随机选择一个区间
        #x_range = random.choice(self.cfg.ranges.pos_x)  # 从两个区间中随机选择一个

        # 使用选中的区间生成随机值
        #self.pos_command_w[env_ids, 0] += r.uniform_(*x_range)  # 解包区间并传给uniform_

        # 随机选择一个区间
        #y_range = random.choice(self.cfg.ranges.pos_y)  # 从两个区间中随机选择一个

        # 使用选中的区间生成随机值
        #self.pos_command_w[env_ids, 1] += r.uniform_(*y_range)  # 解包区间并传给uniform_
        curriculum_threshold = 500 * self.env.num_envs * 8

        if self.env.common_step_counter < curriculum_threshold:  # 第一阶段：极近距离
            pos_x_range = (-0.6, -0.5)
            pos_y_range = (-0.45, -0.35)
        else:                   # 第二阶段：目标距离
            pos_x_range = (-0.9, -0.7)
            pos_y_range = (-0.7, -0.1)


        self.pos_command_w[env_ids, 0] += r.uniform_(*pos_x_range)
        self.pos_command_w[env_ids, 1] += r.uniform_(*pos_y_range)


        # self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        # self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_w[env_ids, 2] += 0.1 #self.robot.data.default_root_state[env_ids, 2]
        self.rover_pos[env_ids, 0] += 0.0
        self.rover_pos[env_ids, 1] += -1.2
        self.rover_pos[env_ids, 2] = 0.15  #0 #0.15

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        #head_pos = self.robot.data.body_pos_w[:,1]
        #target_vec = self.pos_command_w - head_pos[:, :3]
        #self.pos_command_b[:] = quat_rotate_inverse(f(self.robot.data.root_quat_w), target_vec)
        #head_quat = self.robot.data.body_quat_w[:,1]
        #self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(head_quat), target_vec)
        #self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        # 1. 获取头部的位置（只取x, y坐标）
        head_pos = self.robot.data.body_pos_w[:, 0]  # 只获取头部位置的x, y分量
        
        # 2. 计算目标点与头部之间的位置差（只取x, y坐标）
        target_vec = self.pos_command_w - head_pos[:, :3]  # 目标点的位置 - 头部位置
        
        # 3. 获取头部的朝向四元数
        head_quat = self.robot.data.body_quat_w[:, 0]
        
        # 获取头部朝向的前进向量（在XY平面内）
        head_forward_vector = math_utils.quat_apply(head_quat, self.robot.data.FORWARD_VEC_B)
        #head_forward_vector_2d = head_forward_vector[:, :2]  # 只保留x, y分量
        
        # 4. 计算目标点方向与头部朝向之间的夹角（仅在2D平面中）
        dot_product = torch.sum(target_vec * head_forward_vector, dim=1)  # 计算点积
        target_vec_norm = torch.norm(target_vec, dim=1)  # 目标向量的模
        head_forward_vector_norm = torch.norm(head_forward_vector, dim=1)  # 头部前进向量的模
        
        # 计算夹角的余弦值，避免数值误差（限制在[-1, 1]）
        cos_angle = torch.clamp(dot_product / (target_vec_norm * head_forward_vector_norm), -1.0, 1.0)
        
        # 计算夹角（单位：弧度）
        angle = torch.acos(cos_angle)
        
        # 5. 计算目标点相对于头部的方向指令
        # 将目标位置转换到头部的坐标系下（只保留x, y坐标）
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(head_quat), target_vec)  # 只保留x, y分量

        # 6. 计算头部的朝向指令（相对于目标点）
        self.heading_command_b[:] = wrap_to_pi(angle)  # 夹角转换为[-pi, pi]范围内


    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=torch.cat((self.pos_command_w, self.rover_pos), dim=0),
            orientations=quat_from_euler_xyz(
                torch.cat((torch.zeros_like(self.heading_command_w), torch.zeros_like(self.heading_command_w)), dim=0),
                torch.cat((torch.zeros_like(self.heading_command_w), torch.zeros_like(self.heading_command_w)), dim=0),
                torch.cat((self.heading_command_w, self.heading_command_w), dim=0),
            ),
            marker_indices = [0] * self.num_envs + [1] * self.num_envs
        )
        #self.goal_pose_visualizer.visualize(
         #   translations=self.pos_command_w,
          #  orientations=quat_from_euler_xyz(
           #     torch.zeros_like(self.heading_command_w),
            #    torch.zeros_like(self.heading_command_w),
           #     self.heading_command_w,
           # ),
        #)
