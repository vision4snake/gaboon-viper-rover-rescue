import re
import csv
import os
import glob

def extract_robot_data_batch(input_dir, output_dir):
    """
    批量从指定目录下的所有txt文件中提取数据，生成对应的CSV文件
    
    参数:
    input_dir: 存放txt文件的目录路径
    output_dir: 存放生成CSV文件的目录路径
    """
    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 获取目录下所有的 txt 文件
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not txt_files:
        print(f"在 {input_dir} 中没有找到任何 .txt 文件。")
        return

    headers = [
        'Snake_X', 'Snake_Y', 'Snake_Z',
        'Stone_X', 'Stone_Y', 'Stone_Z',
        'Current_Time'
    ]

    print(f"开始处理，共发现 {len(txt_files)} 个文件...")

    for txt_file_path in txt_files:
        # 获取文件名（不含路径和后缀），用于构造输出文件名
        file_name = os.path.splitext(os.path.basename(txt_file_path))[0]
        csv_output_path = os.path.join(output_dir, f"{file_name}.csv")
        
        data_list = []
        
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line: continue
                    
                    # 正则提取逻辑
                    snake_match = re.search(r'Snake Position: \[([^\]]+)\]', line)
                    stone_match = re.search(r'Stone Position: \[([^\]]+)\]', line)
                    time_match = re.search(r'Current Time: ([\d\.]+)', line)
                    
                    if all([snake_match, stone_match, time_match]):
                        # 转换数据
                        snake_coords = [float(v) for v in snake_match.group(1).split()[:3]]
                        stone_coords = [float(v) for v in stone_match.group(1).split()[:3]]
                        current_time = float(time_match.group(1))
                        
                        data_list.append(snake_coords + stone_coords + [current_time])

            # 写入当前文件的 CSV
            if data_list:
                with open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(headers)
                    writer.writerows(data_list)
                print(f"成功: {file_name}.txt -> {file_name}.csv ({len(data_list)}条)")
            else:
                print(f"跳过: {file_name}.txt (无有效数据)")

        except Exception as e:
            print(f"错误: 处理文件 {txt_file_path} 时出错: {e}")

    print("\n所有文件处理完毕！")

# ------------------- 主程序 -------------------
if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "test_csv"      # 原始txt文件所在的文件夹
    OUTPUT_DIR = "test_csv_2"   # 转换后csv文件存放的文件夹
    
    extract_robot_data_batch(INPUT_DIR, OUTPUT_DIR)
