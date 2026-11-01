"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import sys
sys.path.append(r'D:\isaac\IsaacLab-main\source\extensions\omni.isaac.lab')
import argparse
from omni.isaac.lab import __version__ as omni_isaac_lab_version
from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
if omni_isaac_lab_version < "0.21.0":
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from isaacLab.manipulation.algorithms.rsl_rl.rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

# Import extensions to set up environment tasks
import isaacLab.manipulation.tasks  # noqa: F401  TODO: import lab.<your_extension_name>


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    if omni_isaac_lab_version < "0.21.0":
        env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    else:
        env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)#log_root_path, '2026-03-31_21-35-09','model_500.pt'#log_root_path,'2025-11-17_22-23-48','model_1500.pt' #(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)#(r"logs\rsl_rl\snake_robot_navigation\2025-08-07_02-06-05_daice\model_1499.pt")#(r'D:\isaac\IsaacLab-main\isaacLab.manipulation\logs\rsl_rl\snake_robot_navigation\2025-08-04_01-44-20_SOTA\model_1499.pt') #'D:\\isaacsim\\IsaacLab-main\\isaacLab.manipulation\\logs\\rsl_rl\\snake_robot_navigation\\2024-12-10_20-07-48\\model_1499.pt'
    #resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    #D:\isaac\IsaacLab-main\isaacLab.manipulation\logs\rsl_rl\snake_robot_navigation\2025-08-07_02-06-05_daice\model_1499.pt
    
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # root_pos = obs[2, 3:6].cpu().numpy()
            # pos_command = obs[2, :3].cpu().numpy()
            # with open('rosbag_log.txt', 'a') as log_file:  # 使用 'a' 模式以追加内容
            #     log_file.write(f"Snake Position: {root_pos}, Stone Position: {pos_command}, ")
            for i in range(1, 11):  # 循环遍历 1 到 10 的环境
                root_pos = obs[i-1, 3:6].cpu().numpy()  # 获取第 i 个环境的 root_pos
                pos_command = obs[i-1, :3].cpu().numpy()  # 获取第 i 个环境的 pos_command
                
                # 动态生成文件名，例如 'rosbag_log_1', 'rosbag_log_2' 等
                log_filename = f'rosbag_log_{i}.txt'

                # 将数据写入对应的文件
                with open(log_filename, 'a') as log_file:  # 使用 'a' 模式以追加内容
                    log_file.write(f"Snake Position: {root_pos}, Stone Position: {pos_command}, ")
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
