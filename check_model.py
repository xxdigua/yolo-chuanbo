import torch
from models.yolov8 import yolov8_n

# 加载模型
model = yolov8_n()
model.train()

# 创建输入张量
input = torch.randn(1, 3, 640, 640)

# 前向传播
outputs = model(input)

# 检查输出形状
print("Model outputs:")
for i, output in enumerate(outputs):
    print(f"Output {i} shape: {output.shape}")
    # 计算类别通道数
    channels = output.shape[1]
    reg_channels = 4 * 16  # reg_max=16
    cls_channels = channels - reg_channels
    print(f"  Channels: {channels}, Reg channels: {reg_channels}, Cls channels: {cls_channels}")

# 检查模型配置
print("\nModel configuration:")
print(f"Number of classes: {model.head.nc}")
print(f"Reg max: {model.head.reg_max}")
print(f"Number of detection layers: {model.head.nl}")

# 检查每个检测头的输出通道数
print("\nDetection head outputs:")
for i, (cv2, cv3) in enumerate(zip(model.head.cv2, model.head.cv3)):
    # 检查 cv2 的输出通道数
    cv2_output = cv2(torch.randn(1, 64, 80, 80))  # 假设输入通道数为64
    print(f"CV2 {i} output shape: {cv2_output.shape}")
    # 检查 cv3 的输出通道数
    cv3_output = cv3(torch.randn(1, 64, 80, 80))  # 假设输入通道数为64
    print(f"CV3 {i} output shape: {cv3_output.shape}")
