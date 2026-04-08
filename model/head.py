import torch
import torch.nn as nn
import copy
import math
from models.layers import Conv, DFL
from utils.ops import dist2bbox, make_anchors

class Detect(nn.Module):
    def __init__(self, nc=6, ch=()):
        super().__init__()
        self.nc = nc  
        self.nl = len(ch)  
        self.reg_max = 16  
        self.no = nc + self.reg_max * 4  
        self.stride = torch.tensor([8.0, 16.0, 32.0])  
        self.shape = None
        self.anchors = torch.empty(0)  
        self.strides = torch.empty(0)  
        
        # 构建解耦头 (Decoupled Head)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], 100)
        
        # 回归分支
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        # 分类分支
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        # 1. 统一计算卷积输出，只做一次！
        # 结果 res 是一个列表，包含三个尺度的原始预测 [B, 64+nc, H, W]
        res = []
        for i, xi in enumerate(x):
            # 这里 xi 是来自 Backbone/FPN 的原始特征
            res.append(torch.cat((self.cv2[i](xi), self.cv3[i](xi)), 1))
            
        if self.training:
            return res
            
        # 2. 推理模式：直接处理已经卷积好的结果，不要再次调用 self.cv2[i]
        return self._inference(res)

    def _inference(self, x_cat_list):
        shape = x_cat_list[0].shape  
        batch_size = shape[0]
        device = x_cat_list[0].device
        reg_max = self.reg_max
        
        # ====== 与 loss.py 完全一致的解码逻辑 ======
        all_pred_reg, all_pred_cls, all_anchors, all_strides = [], [], [], []
        
        for i, xi in enumerate(x_cat_list):
            b, c, h, w = xi.shape
            stride = self.stride[i].item()
            
            # 与 loss.py 相同的 permute 和 reshape
            pred = xi.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            
            # 分离 reg 和 cls（与 loss.py 一致）
            pred_reg = pred[..., :reg_max * 4].view(b, -1, 4, reg_max)  # [B, N, 4, 16]
            pred_cls = pred[..., reg_max * 4:]  # [B, N, nc]
            
            all_pred_reg.append(pred_reg)
            all_pred_cls.append(pred_cls)
            
            # 生成锚点（与 loss.py 完全一致）
            gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            anchor_points = torch.stack([gx, gy], dim=-1).float() + 0.5
            all_anchors.append((anchor_points * stride).view(-1, 2))  # 像素坐标
            all_strides.append(torch.full((h * w, 1), stride, device=device))
        
        # 合并（与 loss.py 一致）
        all_pred_reg = torch.cat(all_pred_reg, dim=1)   # [B, N, 4, 16]
        all_pred_cls = torch.cat(all_pred_cls, dim=1)     # [B, N, nc]
        all_anchors = torch.cat(all_anchors, dim=0)       # [N, 2] 像素坐标
        all_strides = torch.cat(all_strides, dim=0)       # [N, 1]
        
        # DFL 解码（与 loss.py 第186-196行完全一致）
        proj = torch.arange(reg_max, dtype=torch.float, device=device)
        pd_dist = (torch.softmax(all_pred_reg, dim=-1) * proj).sum(-1)  # [B, N, 4]
        
        anc_points_batch = all_anchors.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, 2]
        strides_batch = all_strides.unsqueeze(0)  # [1, N, 1]
        
        # 计算预测框 (LTRB 格式，像素坐标)
        pd_boxes = torch.cat([
            anc_points_batch - pd_dist[:, :, :2] * strides_batch,
            anc_points_batch + pd_dist[:, :, 2:] * strides_batch
        ], dim=-1)  # [B, N, 4]
        
        # 分类输出 (不加 sigmoid)
        cls_out = all_pred_cls  # [B, N, nc]

        return torch.cat((pd_boxes, cls_out), 2)  # [B, N, 4+nc]

    def decode_bboxes(self, bboxes, anchors):
        """始终返回 LTRB (x1, y1, x2, y2) 格式"""
        # 注意：dist2bbox 内部已经处理了 permute，返回形状 [B, 8400, 4]
        return dist2bbox(bboxes, anchors, xywh=False, dim=-1)

    def bias_init(self):
        """初始化偏置，加速收敛"""
        prior_prob = 0.01
        init_bias = -math.log((1 - prior_prob) / prior_prob)
        for a, b in zip(self.cv2, self.cv3):
            # 回归分支：使用较小初始值，防止 DFL 距离预测爆炸
            a[-1].bias.data[:] = 0.0  # 从 1.0 改为 0.0
            # 同时缩小回归分支最后层的权重
            nn.init.normal_(a[-1].weight, std=0.01)
            
            # 分类分支
            b[-1].bias.data.fill_(init_bias)
            nn.init.normal_(b[-1].weight, std=0.01)
        print(f"✅ Detection Head Bias Initialized: cls={init_bias:.2f}, reg=0.00")