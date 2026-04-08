import torch.nn as nn
from .layers import Conv, C2f, Upsample, Concat

class FPN_PAN(nn.Module):
    """FPN和PAN结构，用于特征融合"""
    def __init__(self, depth_multiple=1.0, width_multiple=1.0):
        super().__init__()
        # 通道数配置
        c3 = int(256 * width_multiple)
        c4 = int(512 * width_multiple)
        c5 = int(1024 * width_multiple)
        
        # 从上到下的 FPN 路径
        self.up1 = Upsample(2)
        self.concat1 = Concat(1)
        self.csp1 = C2f(c4 + c5, c4, n=int(3 * depth_multiple), shortcut=False)
        
        self.up2 = Upsample(2)
        self.concat2 = Concat(1)
        self.csp2 = C2f(c3 + c4, c3, n=int(3 * depth_multiple), shortcut=False)
        
        # 从下到上的 PAN 路径
        self.down1 = Conv(c3, c3, 3, 2, 1)
        self.concat3 = Concat(1)
        self.csp3 = C2f(c3 + c4, c4, n=int(3 * depth_multiple), shortcut=False)
        
        self.down2 = Conv(c4, c4, 3, 2, 1)
        self.concat4 = Concat(1)
        self.csp4 = C2f(c4 + c5, c5, n=int(3 * depth_multiple), shortcut=False)
    
    def forward(self, x1, x2, x3, x4):#（160*160）（80*80）（40*40）（20*20）
        """前向传播"""
        # FPN 路径
        p5 = x4
        p4 = self.csp1(self.concat1([self.up1(p5), x3]))#20*20经过上采样与40*40进行concat再经过C2F_2_3输出40*40*512
        p3 = self.csp2(self.concat2([self.up2(p4), x2]))#P4经过上采样与80*80进行concat再经过C2F_2_3输出80*80*256
        
        # PAN 路径
        p4 = self.csp3(self.concat3([self.down1(p3), p4]))#P3经过CBS与P4进行concat再经过C2F_2_3输出40*40*512
        p5 = self.csp4(self.concat4([self.down2(p4), p5]))#P4经过CBS与P5进行concat再经过C2F_2_3输出20*20*1024
        
        return p3, p4, p5#（80*80*256）（40*40*512）（20*20*1024）
