import xml.etree.ElementTree as ET

def modify_urdf(input_file, output_file):
    # 解析 URDF 文件
    tree = ET.parse(input_file)
    root = tree.getroot()

    # 遍历所有 joint 元素
    for joint in root.findall('joint'):
        # 只修改类型为 continuous 的 joint
        if joint.get('type') == 'continuous':
            # 将类型改为 revolute
            joint.set('type', 'revolute')

            # 查找并添加 joint 限制元素
            limit = ET.Element('limit')
            limit.set('lower', str(-60 * 3.14159 / 180))  # -15 度
            limit.set('upper', str(60 * 3.14159 / 180))   # 15 度

            # 将 limit 元素添加到 joint 中
            joint.append(limit)

    # 将修改后的 URDF 写回新文件
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# 使用示例
input_urdf = "D:\\isaacsim\\IsaacLab-main\\isaacLab.manipulation\\isaacLab\\manipulation\\assets\\urdf\\snake_robot\\urdf\\snake_robot.urdf"
output_urdf = "D:\\isaacsim\\IsaacLab-main\\isaacLab.manipulation\\isaacLab\\manipulation\\assets\\urdf\\snake_robot\\urdf\\modified_snake_robot.urdf"
modify_urdf(input_urdf, output_urdf)

print(f"URDF has been modified and saved to {output_urdf}")

