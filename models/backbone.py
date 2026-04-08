import torch.nn as nn
from .layers import Conv, C2f, SPPF

class CSPDarknet(nn.Module):
    """CSPDarknet 主干网络"""
    def __init__(self, depth_multiple=1.0, width_multiple=1.0):
        super().__init__()
        # 初始卷积
        self.stem = Conv(3, int(64 * width_multiple), 3, 2, 1)#320*320*64
        # 第1阶段
        self.stage1 = nn.Sequential(
            Conv(int(64 * width_multiple), int(128 * width_multiple), 3, 2, 1),#160*160*128
            C2f(int(128 * width_multiple), int(128 * width_multiple), n=int(3 * depth_multiple))#160*160*128
        )
        # 第2阶段
        self.stage2 = nn.Sequential(
            Conv(int(128 * width_multiple), int(256 * width_multiple), 3, 2, 1),#80*80*256
            C2f(int(256 * width_multiple), int(256 * width_multiple), n=int(6 * depth_multiple))#80*80*256
        )
        # 第3阶段
        self.stage3 = nn.Sequential(
            Conv(int(256 * width_multiple), int(512 * width_multiple), 3, 2, 1),#40*40*512
            C2f(int(512 * width_multiple), int(512 * width_multiple), n=int(6 * depth_multiple))#40*40*512
        )
        # 第4阶段
        self.stage4 = nn.Sequential(
            Conv(int(512 * width_multiple), int(1024 * width_multiple), 3, 2, 1),#20*20*1024
            C2f(int(1024 * width_multiple), int(1024 * width_multiple), n=int(3 * depth_multiple)),#20*20*1024
            SPPF(int(1024 * width_multiple), int(1024 * width_multiple))#20*20*1024
        )
    
    def forward(self, x):
        """前向传播，返回不同尺度的特征图"""
        x = self.stem(x)
        x1 = self.stage1(x)    # 1/4 尺度 160*160*128
        x2 = self.stage2(x1)   # 1/8 尺度 80*80*256
        x3 = self.stage3(x2)   # 1/16 尺度 40*40*512
        x4 = self.stage4(x3)   # 1/32 尺度 20*20*1024
        return x1, x2, x3, x4