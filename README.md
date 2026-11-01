# Gaboon viper rover rescue — release preparation

Private target: https://github.com/vision4snake/gaboon-viper-rover-rescue.git

## Environment evidence

`environment.yml` is a full Windows snapshot exported from `env_isaaclab` (Python 3.10.13), with the machine-specific prefix removed. It is not a minimal or validated installation specification. Create with `conda env create -f environment.yml`; environment resolution has not been tested. Local editable packages and simulator-provided extensions need separate installation.

Important: the Conda environment contains Isaac Sim 4.5.0.0, whereas D:/isaac/isaacsim/VERSION reports 4.2.0-rc.18+release.16044.3b2ed111.gl. The supplied python.bat defaults to the simulator bundled interpreter unless PYTHONEXE is set. Activating Conda alone does not select its Python for this command. requirements-isaacsim-bundled.txt inventories packages visible to that bundled Python without simulator startup; runtime extensions are additional dependencies.

The copied parent Isaac Lab source reports version 1.2.0. It is included under isaaclab-source because navigation imports customized parent modules, rather than exclusively the extension's local modules. Original licenses are retained.

## Author-provided commands (original workstation)

Run from the original extension directory; these are recorded for provenance, not executed during packaging:

```bat
conda activate env_isaaclab
D:\isaac\isaacsim\python scripts/rsl_rl/play.py --task Isaac-Navigation-Flat-Snake-Robot-Play-v0 --num_envs 64
D:\isaac\isaacsim\python scripts/rsl_rl/train.py --task Isaac-Navigation-Flat-Snake-Robot-v0 --num_envs 4096 --headless
```

Task configuration: code/isaacLab/manipulation/tasks/Snake_robot/navigation/config/snake_robot/navigation_env_cfg.py.
Algorithm: locally bundled RSL-RL PPO; retain the bundled implementation rather than silently replacing it with a pip release.

## Terrain

terrain/ contains procedural generation Python sources and the original CC0 license. Generated assets have not been bundled. Author parameters: Seed=922, Scale=[5.0,4.0,0.25], Density=0.1, Flat Area Size=1.0, Rock Mesh Boolean=false; material MartianSurface; texture_resolution=4096; seed=922; num_assets=500; ext=.usd; output directory outputs_20250804. Use scripts/blender/procgen_assets.py with model/terrain/martian_surface_procgen/geometry.py and material.py as autorun scripts in Blender.

Current training inherits a plane terrain. Play explicitly uses outputs_eval2/combined.usd. Generating outputs_20250804 is not by itself sufficient to reproduce that evaluation scene: asset selection/composition and referenced USD/textures must also be preserved.

## Remaining release work

This is an archival staging copy, not yet a portable validated release. Resolve absolute paths in the copy, verify the active runtime and checkpoint, collect the evaluation scene's referenced assets, check third-party license provenance, restore any required TXT resources omitted during initial filtering, and run a small simulation smoke test before publication. Do not launch the full 4096-environment training or 500-terrain generation as an installation check.

No source files in the original project were edited. manifest.csv covers the initial extension copy only; newly included framework and terrain files are not covered by that initial manifest. No GitHub upload has been completed.
