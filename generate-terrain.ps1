param([string]$Blender = 'D:\downloads\blender.exe', [string]$OutputDirectory = "$PSScriptRoot\outputs_20250804")
$ErrorActionPreference = 'Stop'
$settings = '{"MartianTerrain":{"Seed":922,"Scale":[5.0,4.0,0.25],"Density":0.1,"Flat Area Size":1.0,"Rock Mesh Boolean":false}}'
# Windows native argument quoting for Blender's JSON argument.
$jsonArg = $settings.Replace('"', '\"')
& $Blender --factory-startup --background --offline-mode --enable-autoexec --python "$PSScriptRoot\terrain\scripts\blender\procgen_assets.py" -- --autorun_scripts "$PSScriptRoot\terrain\model\terrain\martian_surface_procgen\geometry.py" "$PSScriptRoot\terrain\model\terrain\martian_surface_procgen\material.py" --geometry_nodes $jsonArg --material MartianSurface --texture_resolution 4096 --seed 922 --num_assets 500 --ext '.usd' --outdir $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Blender terrain generation failed.' }
