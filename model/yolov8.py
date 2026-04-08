import torch
import torch.nn as nn
from .backbone import CSPDarknet
from .neck import FPN_PAN
from .head import Detect

class YOLOv8(nn.Module):
    """YOLOv8 完整模型"""
    def __init__(self, num_classes=6, depth_multiple=1.0, width_multiple=1.0, reg_max=16):
        super().__init__()
        self.backbone = CSPDarknet(depth_multiple, width_multiple)
        self.neck = FPN_PAN(depth_multiple, width_multiple)
        # 计算通道数
        c3 = int(256 * width_multiple)
        c4 = int(512 * width_multiple)
        c5 = int(1024 * width_multiple)
        self.head = Detect(nc=num_classes, ch=[c3, c4, c5])
        # 设置步长
        self.register_buffer('stride', torch.tensor([8.0, 16.0, 32.0]))
        self.head.stride = self.stride
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        x1, x2, x3, x4 = self.backbone(x)# 主干网络进行特征提取输出特征图
        # 特征融合
        p3, p4, p5 = self.neck(x1, x2, x3, x4)# 颈部融合特征图输出
        # 预测输出
        outputs = self.head([p3, p4, p5])# 头部解耦头预测输出
        return outputs

# 模型配置
def yolov8_n(num_classes=6):
    """YOLOv8n 模型"""
    return YOLOv8(num_classes=num_classes, depth_multiple=0.33, width_multiple=0.25)

def yolov8_s(num_classes=80):
    """YOLOv8s 模型"""
    return YOLOv8(num_classes=num_classes, depth_multiple=0.33, width_multiple=0.5)

def yolov8_m(num_classes=80):
    """YOLOv8m 模型"""
    return YOLOv8(num_classes=num_classes, depth_multiple=0.67, width_multiple=0.75)

def yolov8_l(num_classes=80):
    """YOLOv8l 模型"""
    return YOLOv8(num_classes=num_classes, depth_multiple=1.0, width_multiple=1.0)

def yolov8_x(num_classes=80):
    """YOLOv8x 模型"""
    return YOLOv8(num_classes=num_classes, depth_multiple=1.33, width_multiple=1.25)