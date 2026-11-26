# Gaboon Viper Rover Rescue

Research code for snake-robot reinforcement learning and procedural terrain generation associated with the Gaboon Viper Rover Rescue project.

## Repository structure

- `code/`: robot assets, task configurations, and training and evaluation scripts, including the bundled RSL-RL implementation.
- `isaaclab-source/`: Isaac Lab framework source used by the project, including customized modules imported by the navigation task.
- `terrain/`: procedural terrain generation source and its original license.
- `environment.yml`: Windows Conda environment snapshot.
- `requirements-isaacsim-bundled.txt`: package inventory from the local Isaac Sim bundled Python interpreter.
- `generate-terrain.ps1`: PowerShell script for terrain generation.

## Software environment

The project uses Python 3.10 and NVIDIA Isaac Sim with Isaac Lab. The framework source in this repository reports Isaac Lab version 1.2.0.

The recorded Conda environment uses Python 3.10.13, PyTorch 2.5.1+cu121, and NumPy 1.26.4. To create the recorded environment:

```bat
conda env create -f environment.yml
conda activate env_isaaclab
```

The environment file records installed packages; Isaac Sim runtime extensions and local project packages require separate configuration. Use the bundled RSL-RL implementation for this project's PPO training.

### Simulator interpreter

The original workstation's simulator launcher is `D:\isaac\isaacsim\python.bat`, with Isaac Sim version `4.2.0-rc.18+release.16044.3b2ed111.gl`. The Conda snapshot also contains Isaac Sim 4.5.0.0 packages. These refer to separate installations: the supplied launcher defaults to Isaac Sim's bundled Python unless `PYTHONEXE` is set. Activating Conda does not by itself change that interpreter.

## Local configuration

Before running, configure source and asset paths for your installation. The source retains workstation-specific paths for the Isaac Lab extensions, robot USD assets, and terrain. Point the extension imports to `isaaclab-source/extensions/` and the robot assets to their locations under `code/`. Keep the customized framework modules on the Python import path.

Run the following commands from `code/`, replacing the simulator launcher path with your local installation path.

## Training

```bat
conda activate env_isaaclab
D:\isaac\isaacsim\python scripts/rsl_rl/train.py --task Isaac-Navigation-Flat-Snake-Robot-v0 --num_envs 4096 --headless
```

The navigation task uses the bundled RSL-RL PPO implementation. Its configuration is located at:

```text
code/isaacLab/manipulation/tasks/Snake_robot/navigation/config/snake_robot/navigation_env_cfg.py
```

The current training configuration inherits a plane terrain.

## Evaluation

```bat
conda activate env_isaaclab
D:\isaac\isaacsim\python scripts/rsl_rl/play.py --task Isaac-Navigation-Flat-Snake-Robot-Play-v0 --num_envs 64
```

Evaluation requires a trained checkpoint in the configured log directory. Configure the run and checkpoint selection for the model to evaluate. Training logs and model checkpoints are stored separately from the source repository.

The evaluation configuration refers to `outputs_eval2/combined.usd`. Set this path to the evaluation scene on your machine and ensure its referenced USD assets and textures are available.

## Procedural terrain generation

The terrain generator uses Blender with `procgen_assets.py`, loading `geometry.py` and `material.py` from `terrain/model/terrain/martian_surface_procgen/`.

From the repository root, run:

```powershell
.\generate-terrain.ps1 -Blender 'D:\downloads\blender.exe' -OutputDirectory 'D:\terrain\generated_terrain'
```

The script uses these parameters:

| Parameter | Value |
| --- | --- |
| Geometry node group | `MartianTerrain` |
| Geometry seed | 922 |
| Scale | [5.0, 4.0, 0.25] |
| Density | 0.1 |
| Flat area size | 1.0 |
| Rock mesh boolean | false |
| Material | `MartianSurface` |
| Texture resolution | 4096 |
| Generation seed | 922 |
| Number of assets | 500 |
| Output format | `.usd` |

Generated terrain assets are stored separately. The generation output `generated_terrain` and the composed evaluation scene `outputs_eval2/combined.usd` serve different purposes; evaluation also requires the scene composition and its referenced assets.

## Licenses

Original license files and source copyright notices are retained. See `ISAACLAB-LICENSE`, the licenses within `code/`, and `terrain/LICENSE-CC0` for the corresponding components.
