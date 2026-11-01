# 输入和输出文件路径
input_file = 'action_log.txt'  # 你的原始日志文件路径
output_file = 'processed_log.txt'  # 处理后的日志文件路径

# 打开输入文件读取内容
with open(input_file, 'r') as file:
    lines = file.readlines()

# 处理每一行数据
processed_lines = []
for line in lines:
    # 查找 "Root Position: tensor([" 和 "Current Time:" 的位置
    start_pos = line.find("tensor([")
    end_pos = line.find("], device='cuda:0')")
    time_pos = line.find("Current Time:")

    if start_pos != -1 and end_pos != -1 and time_pos != -1:
        # 提取 "tensor([" 和 "], device='cuda:0')" 之间的部分，即坐标部分
        coords_str = line[start_pos + len("tensor(["):end_pos]
        # 提取 "Current Time:" 后的时间部分
        time_str = line[time_pos + len("Current Time:"):].strip()

        # 去除坐标字符串中的多余空格并分割成两个坐标值
        coords = coords_str.strip().split(",")
        x = coords[0].strip()
        y = coords[1].strip()

        # 构建新的行并添加到处理后的列表
        processed_line = f"{x} {y} {time_str}\n"
        processed_lines.append(processed_line)

# 将处理后的数据写入到新文件
with open(output_file, 'w') as file:
    file.writelines(processed_lines)

print(f"处理完成，结果已保存到 {output_file}")

