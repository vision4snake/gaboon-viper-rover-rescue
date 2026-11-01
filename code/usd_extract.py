import os
import glob
import numpy as np
import pandas as pd
from scipy.io import savemat
from pxr import Usd, UsdGeom

# ================= 配置区域 =================
# 输入：你的 USD 文件所在的文件夹路径
USD_FOLDER_PATH = r"D:\isaac\srb_assets-main\srb_assets-main\srb_assets-main\scripts\blender\outputs_20250804" 

# 输出：保存结果的文件名
OUTPUT_CSV = "terrain_data.csv"
OUTPUT_MAT = "terrain_data.mat"
# ===========================================

def extract_metrics_from_usd(file_path):
    """
    读取单个 USD 文件，返回 (粗糙度, 坡度)
    """
    try:
        # 打开 USD 场景
        stage = Usd.Stage.Open(file_path)
        if not stage:
            print(f"Error: 无法打开 {file_path}")
            return None

        all_points = []
        
        # 遍历场景中的所有 Mesh
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                # 获取顶点坐标属性
                points_attr = mesh.GetPointsAttr().Get()
                if points_attr:
                    pts = np.array(points_attr)
                    if len(pts) > 0:
                        all_points.append(pts)

        if not all_points:
            print(f"Warning: {file_path} 中没有找到有效的 Mesh 数据")
            return None

        # 合并所有 Mesh 顶点
        # 假设地形由多个 Mesh 组成，或者就是一个大 Mesh
        points_combined = np.vstack(all_points)
        
        # --- 计算物理指标 ---
        # 假设 Z 轴 (索引2) 是高度轴。如果是 Y 轴向上，请改为 points_combined[:, 1]
        z_values = points_combined[:, 2]
        
        # 1. 粗糙度 (Roughness) = Z 值的标准差
        roughness = np.std(z_values)
        
        # 2. 近似最大坡度 (Approx Slope)
        # 这里使用一种鲁棒的估计方法：最大高差 / 地形对角线长度 的反正切
        # 或者更简单：直接用最大高差作为"陡峭度"的代理指标
        # 也可以计算局部法向量，但比较耗时。这里用高差代替坡度特征用于分布分析是足够的。
        max_height_diff = np.max(z_values) - np.min(z_values)
        
        # 如果你想算得更像"度数"，可以假设地形是 10m x 10m
        # slope_deg = np.degrees(np.arctan(max_height_diff / 10.0))
        # 这里直接返回最大高差
        slope_metric = max_height_diff 

        return roughness, slope_metric

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # 获取所有 USD 文件
    usd_files = glob.glob(os.path.join(USD_FOLDER_PATH, "*.usd")) + \
                glob.glob(os.path.join(USD_FOLDER_PATH, "*.usda")) + \
                glob.glob(os.path.join(USD_FOLDER_PATH, "*.usdc"))
    
    print(f"找到 {len(usd_files)} 个 USD 文件，开始处理...")
    
    results = []
    
    for i, file_path in enumerate(usd_files):
        print(f"[{i+1}/{len(usd_files)}] 处理: {os.path.basename(file_path)} ...", end="\r")
        
        metrics = extract_metrics_from_usd(file_path)
        
        if metrics:
            r, s = metrics
            results.append({
                "Filename": os.path.basename(file_path),
                "Roughness": r,
                "Slope": s
            })
    
    print(f"\n处理完成！成功提取 {len(results)} 个文件的数据。")
    
    if len(results) == 0:
        return

    # --- 1. 保存为 CSV (通用格式) ---
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"数据已保存至 CSV: {os.path.abspath(OUTPUT_CSV)}")
    
    # --- 2. 保存为 MAT (MATLAB 专用) ---
    # 将数据转换为 MATLAB 字典格式
    # 结构: data.Roughness (数组), data.Slope (数组), data.Filenames (Cell数组)
    mat_data = {
        "Roughness": df["Roughness"].values,
        "Slope": df["Slope"].values,
        "Filenames": df["Filename"].values
    }
    savemat(OUTPUT_MAT, mat_data)
    print(f"数据已保存至 MAT: {os.path.abspath(OUTPUT_MAT)}")

if __name__ == "__main__":
    main()