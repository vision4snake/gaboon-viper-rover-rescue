from pxr import Usd, UsdGeom
import os

def create_terrain_grid(terrain_dir, output_path='output.usd', grid_size=(3, 3)):
    # 创建新的USD舞台
    stage = Usd.Stage.CreateNew(output_path)

    # 获取地形文件路径
    terrain_files = [os.path.join(terrain_dir, f'model{i}.usd') for i in range(42, 51)]

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            index = i * grid_size[1] + j
            if index < len(terrain_files):
                terrain_path = terrain_files[index]
                
                # 加载每个地形USD文件
                terrain_stage = Usd.Stage.Open(terrain_path)
                
                # 假设地形的根prim路径为"/"
                terrain_prim = terrain_stage.GetPrimAtPath("/")
                
                # 在新舞台中复制地形
                if terrain_prim.IsValid():
                    # 创建一个新的组作为父级
                    group_path = f"/TerrainGrid/Terrain_{index}"
                    group_prim = UsdGeom.Xform.Define(stage, group_path)
                    
                    # 复制地形到新的位置
                    Usd.ModelAPI(group_prim).CreateModel(terrain_prim)
                    # 设置位置为 (j * 10, i * 10, 0)，实现紧密排列
                    group_prim.AddTranslateOp().Set(value=(j * 10, i * 10, 0))  # 每个地形紧密排列

    # 保存新舞台
    stage.GetRootLayer().Save()

# 使用示例
terrain_directory = r'D:\Download\srb_assets-main\srb_assets-main\scripts\blender\outputs_usd'
create_terrain_grid(terrain_directory, output_path=r'D:\Download\srb_assets-main\srb_assets-main\scripts\blender\outputs_usd\terrain_grid.usd')
