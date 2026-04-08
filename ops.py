import torch

def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] -> [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    """
    解码 DFL 输出为边界框坐标。
    distance: (batch, 4, num_anchors) -> 偏移量
    anchor_points: (num_anchors, 2) -> 锚点中心坐标
    """
    # 统一转置为 (batch, num_anchors, 4)
    if distance.dim() == 3 and distance.shape[1] == 4:
        distance = distance.permute(0, 2, 1)
        
    # 如果 anchor_points 是 (N, 2)，扩展为 (1, N, 2)
    if anchor_points.dim() == 2:
        anchor_points = anchor_points.unsqueeze(0)

    # 计算左上角(lt)和右下角(rb)
    # distance[..., :2] 是左、上偏移；distance[..., 2:] 是右、下偏移
    lt = anchor_points - distance[..., :2]
    rb = anchor_points + distance[..., 2:]
    
    if xywh:
        # 返回 [cx, cy, w, h]
        return torch.cat([(lt + rb) / 2, rb - lt], dim)
    else:
        # 返回 [x1, y1, x2, y2] -> 推荐使用，直接对接 NMS
        return torch.cat([lt, rb], dim)

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """生成特征图对应的锚点和步长张量"""
    anchor_points, stride_tensor = [], []
    assert len(feats) == len(strides)
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=feats[i].device) + grid_cell_offset
        sy = torch.arange(end=h, device=feats[i].device) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack([sx, sy], dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)