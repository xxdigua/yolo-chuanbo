import os

# 修改为你的 labels 文件夹路径
label_path = "D:/yolo/yolov8/data/VOCdevkit/labels/train" 
num_classes = 19

def check_labels(path):
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    for file in files[:10]:  # 检查前10个文件
        with open(os.path.join(path, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                cls = float(data[0])
                # 检查点 1：类别 ID 是否为整数
                if not cls.is_integer():
                    print(f"错误: 文件 {file} 中的 class_id 是小数: {cls}")
                # 检查点 2：类别 ID 是否在 0 到 18 之间
                if cls >= num_classes:
                    print(f"错误: 文件 {file} 中的 class_id ({int(cls)}) 超过了定义的类别数 {num_classes}")
                # 检查点 3：坐标是否归一化（0-1之间）
                coords = [float(x) for x in data[1:]]
                if any(c > 1.0 for c in coords):
                    print(f"警告: 文件 {file} 中的坐标未归一化（大于1）")

check_labels(label_path)
print("检查完成。")