import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    for i, feat in enumerate(feats):
        _, _, h, w = feat.shape
        sx = torch.arange(end=w, device=feat.device, dtype=feat.dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=feat.device, dtype=feat.dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), strides[i], device=feat.device, dtype=feat.dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class TaskAlignedAssigner:
    """Task-aligned assigner for YOLOv8."""

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
        stride: list = None,
        topk2: Optional[int] = None,
    ):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.topk2 = topk2 if topk2 is not None else topk

    def __call__(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Task-aligned assignment."""
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1) if gt_bboxes.ndim == 3 else gt_bboxes.size(0)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], False).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.full_like(pd_scores[..., 0], False).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        # 计算对齐度指标
        iou = self.iou_calculation(pd_bboxes, gt_bboxes)
        
        # 确保iou和pd_scores形状匹配
        # pd_scores: [batch, anchors, classes]
        # iou: [batch, anchors, gt_boxes]
        
        # 扩展pd_scores到gt_boxes维度
        # 对于每个gt_box，使用对应的类别分数
        align_metric = []
        for i in range(self.bs):
            # 获取当前批次的gt_labels
            gt_label = gt_labels[i, :self.n_max_boxes].long()
            # 获取当前批次的pd_scores
            score = pd_scores[i]
            # 获取当前批次的iou
            iou_i = iou[i]
            
            # 为每个gt_box选择对应的类别分数
            class_scores = score[:, gt_label]  # [anchors, gt_boxes]
            # 计算对齐度指标
            metric = class_scores.pow(self.alpha) * iou_i.pow(self.beta)
            align_metric.append(metric)
        
        align_metric = torch.stack(align_metric, dim=0)

        # 选择topk候选
        topk_metrics, topk_idxs = torch.topk(align_metric, self.topk, dim=-1, largest=True)
        topk_mask = topk_metrics > 0

        # 获取对应的预测和真实值
        batch_idx = torch.arange(self.bs, device=gt_bboxes.device).view(-1, 1)
        target_gt_idx = topk_idxs[topk_mask]

        # 创建目标张量
        target_labels = torch.zeros_like(pd_scores)
        target_bboxes = torch.zeros_like(pd_bboxes)
        fg_mask = torch.zeros_like(pd_scores[..., 0]).bool()

        if target_gt_idx.numel() > 0:
            # 分配标签和边界框
            fg_mask[batch_idx, topk_mask] = True
            target_labels[batch_idx, topk_mask] = gt_labels[batch_idx, target_gt_idx]
            target_bboxes[batch_idx, topk_mask] = gt_bboxes[batch_idx, target_gt_idx]

        return fg_mask, target_bboxes, target_labels, fg_mask, target_gt_idx

    def iou_calculation(self, pd_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between predicted and ground truth boxes."""
        # 确保gt_bboxes在与pd_bboxes相同的设备上
        gt_bboxes = gt_bboxes.to(pd_bboxes.device)
        
        # 确保gt_bboxes是3维的 [B, M, 4]
        if gt_bboxes.dim() == 2:
            gt_bboxes = gt_bboxes.unsqueeze(0)
        
        # 确保pd_bboxes是3维的 [B, N, 4]
        if pd_bboxes.dim() == 2:
            pd_bboxes = pd_bboxes.unsqueeze(0)
        
        # 批量处理
        B, N, _ = pd_bboxes.shape
        B, M, _ = gt_bboxes.shape
        
        # 扩展维度以计算所有预测框与所有真实框的IoU
        pd_bboxes = pd_bboxes.unsqueeze(2)  # [B, N, 1, 4]
        gt_bboxes = gt_bboxes.unsqueeze(1)  # [B, 1, M, 4]
        
        # 计算交集
        lt = torch.max(pd_bboxes[..., :2], gt_bboxes[..., :2])
        rb = torch.min(pd_bboxes[..., 2:], gt_bboxes[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        # 计算并集
        pd_area = (pd_bboxes[..., 2] - pd_bboxes[..., 0]) * (pd_bboxes[..., 3] - pd_bboxes[..., 1])
        gt_area = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])
        union = pd_area + gt_area - inter
        
        iou = inter / (union + 1e-9)
        return iou


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the BboxLoss module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: float,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute box regression loss and DFL loss."""
        # 初始化损失
        loss = torch.zeros(2, device=pred_dist.device)  # [bbox_loss, dfl_loss]

        if fg_mask.sum():
            # IoU loss
            iou = self.iou_calculation(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss[0] = ((1.0 - iou) * target_scores[fg_mask]).sum() / target_scores_sum

            # DFL loss
            target_ltrb = bbox2dist(anchor_points[fg_mask], target_bboxes[fg_mask], self.reg_max)
            loss[1] = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb) * target_scores_sum

        return loss[0], loss[1]

    def _df_loss(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses."""
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = target - tl  # weight right
        
        # 计算DFL损失
        loss_left = F.cross_entropy(pred_dist, tl, reduction='none')
        loss_right = F.cross_entropy(pred_dist, tr, reduction='none')
        return (loss_left * wl + loss_right * wr).mean()

    def iou_calculation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate IoU with CIoU enhancement."""
        # 计算交集
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        # 计算并集
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = pred_area + target_area - inter
        
        iou = inter / (union + 1e-9)
        
        # CIoU计算
        cw = torch.max(pred[:, 2], target[:, 2]) - torch.min(pred[:, 0], target[:, 0])
        ch = torch.max(pred[:, 3], target[:, 3]) - torch.min(pred[:, 1], target[:, 1])
        c2 = cw ** 2 + ch ** 2 + 1e-9
        rho2 = ((pred[:, 0] + pred[:, 2] - target[:, 0] - target[:, 2]) ** 2 +
                (pred[:, 1] + pred[:, 3] - target[:, 1] - target[:, 3]) ** 2) / 4
        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan((pred[:, 2] - pred[:, 0]) / (pred[:, 3] - pred[:, 1] + 1e-9)) -
                                              torch.atan((target[:, 2] - target[:, 0]) / (target[:, 3] - target[:, 1] + 1e-9)), 2)
        alpha = v / (1 - iou + v + 1e-9)
        
        return iou - (rho2 / c2 + v * alpha)