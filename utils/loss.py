import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np

def visualize_tal_assignments(img_bgr, pd_boxes, gt_bboxes, mask_pos, anc_points, save_path="debug.jpg"):
        """
        可视化 TAL 分配结果
        img_bgr: OpenCV 格式图片 (H, W, 3)
        pd_boxes: [N, 4] 预测框 (像素坐标)
        gt_bboxes: [M, 4] 真实框 (像素坐标)
        mask_pos: [M, N] 分配矩阵
        anc_points: [N, 2] 锚点中心
        """
        img = img_bgr.copy()
    # 绘制 GT 框 (绿色)
        for box in gt_bboxes:
            if box.sum() == 0: continue
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # 提取被选中的正样本索引
        pos_indices = torch.where(mask_pos)
        for gt_idx, anc_idx in zip(pos_indices[0], pos_indices[1]):
        # 绘制锚点 (红色小点)
            pt = anc_points[anc_idx]
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        
        # 绘制对应的预测框 (蓝色)
            p_box = pd_boxes[anc_idx]
            cv2.rectangle(img, (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3])), (255, 0, 0), 1)
 
        cv2.imwrite(save_path, img)

class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes=6, reg_max=16, tal_topk=10):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max 
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # 权重系数
        self.bbox_loss_weight = 7.5
        self.cls_loss_weight = 1.0
        self.dfl_loss_weight = 1.5
        
        # TAL 参数
        self.topk = tal_topk 
        self.alpha = 0.5    
        self.beta = 6.0     
    
    

    def bbox_iou(self, box1, box2, CIoU=True, eps=1e-7):
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])
        
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        w1, h1 = (box1[..., 2] - box1[..., 0]).clamp(eps), (box1[..., 3] - box1[..., 1]).clamp(eps)
        w2, h2 = (box2[..., 2] - box2[..., 0]).clamp(eps), (box2[..., 3] - box2[..., 1]).clamp(eps)
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union
        
        if CIoU:
            cw = torch.max(box1[..., 2], box2[..., 2]) - torch.min(box1[..., 0], box2[..., 0])
            ch = torch.max(box1[..., 3], box2[..., 3]) - torch.min(box1[..., 1], box2[..., 1])
            c2 = cw.pow(2) + ch.pow(2) + eps
            rho2 = ((box1[..., 0] + box1[..., 2] - box2[..., 0] - box2[..., 2]) ** 2 +
                    (box1[..., 1] + box1[..., 3] - box2[..., 1] - box2[..., 3]) ** 2) / 4
            v = (4 / torch.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
            with torch.no_grad():
                alpha = v / (v - iou + (1.0 + eps))
            return iou - (rho2 / c2 + v * alpha)
        return iou

    def get_tal_mask(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        bs, n_anchors, _ = pd_scores.shape
        _, n_max_gt, _ = gt_bboxes.shape

        # 1. 计算 IoU [B, M, N]
        # gt_bboxes: [B, M, 4] -> [B, M, 1, 4]
        # pd_bboxes: [B, N, 4] -> [B, 1, N, 4]
        ious = self.bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), CIoU=False)
        
        # 2. 提取预测类别得分 [B, M, N]
        # 找到每个 GT 类别对应的预测分值
        target_labels = gt_labels.long().flatten() # [B*M]
        batch_idx = torch.arange(bs, device=pd_scores.device).view(-1, 1).expand(-1, n_max_gt).flatten()
        
        # 从 pd_scores [B, N, C] 中提取对应类别的得分
        # 结果形状应为 [B, M, N]
        scores = pd_scores[batch_idx, :, target_labels].view(bs, n_max_gt, n_anchors)

        # 3. 计算对齐指标 (Task-Alignment Metric)
        align_metrics = scores.pow(self.alpha) * ious.pow(self.beta)
        
        # 4. 空间约束：锚点必须在 GT 框内 (Center Prior)
        # 计算锚点到 GT 四边的距离 (Left, Top, Right, Bottom)
        # anc_points: [B, N, 2], gt_bboxes: [B, M, 4]
        lt = anc_points.unsqueeze(1) - gt_bboxes[:, :, :2].unsqueeze(2) # [B, M, N, 2]
        rb = gt_bboxes[:, :, 2:].unsqueeze(2) - anc_points.unsqueeze(1) # [B, M, N, 2]
        # 必须全部为正值，才表示锚点在框内
        is_in_gts = torch.cat([lt, rb], dim=-1).amin(-1) > 1e-9 # [B, M, N]
        
        # 结合 GT 掩码和空间掩码
        align_metrics *= is_in_gts 
        align_metrics *= mask_gt.view(bs, n_max_gt, 1)

        # 5. Top-K 选择
        # 为每个 GT 选择前 K 个最匹配的锚点
        topk_metrics, topk_indices = torch.topk(align_metrics, self.topk, dim=-1, largest=True)
        mask_pos = torch.zeros_like(align_metrics, dtype=torch.bool).scatter_(-1, topk_indices, True)
        
        # 过滤掉那些 metric 为 0 的 (即不在框内的)
        mask_pos &= (align_metrics > 0)
        
        # 6. 处理“一个锚点匹配多个 GT”的情况
        if mask_pos.sum(1).max() > 1:
            mask_multi_gts = mask_pos.sum(1) > 1 # [B, N]
            max_metric_idx = align_metrics.argmax(1) # [B, N]
            # 这种情况下，锚点只分配给指标最高的那个 GT
            full_indices = F.one_hot(max_metric_idx, num_classes=n_max_gt).permute(0, 2, 1).bool()
            mask_pos = torch.where(mask_multi_gts.unsqueeze(1), full_indices, mask_pos)
        
        return mask_pos, align_metrics

    def dfl_loss(self, pred_dist, target_ltrb):
        target_ltrb = target_ltrb.clamp(0, self.reg_max - 1.01)
        tl = target_ltrb.long()
        tr = tl + 1
        wl = tr - target_ltrb
        wr = target_ltrb - tl
        l1 = F.cross_entropy(pred_dist.view(-1, self.reg_max), tl.view(-1), reduction='none') * wl.view(-1)
        l2 = F.cross_entropy(pred_dist.view(-1, self.reg_max), tr.view(-1), reduction='none') * wr.view(-1)
        return (l1 + l2).view(target_ltrb.shape).sum(1)

    def xywh2xyxy(self, x):
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def forward(self, outputs, targets):
        device = targets.device
        predictions = outputs 
        bs = predictions[0].shape[0]
        
        # 1. 整理预测分支
        if isinstance(predictions, tuple):
            predictions = predictions[1] if len(predictions) > 1 else predictions[0]
        pred_list = predictions if isinstance(predictions, list) else [predictions]

        all_pred_reg, all_pred_cls, all_anchors, all_strides = [], [], [], []
        strides = [8, 16, 32] 

        for i, pred in enumerate(pred_list):
            if not isinstance(pred, torch.Tensor): continue
            stride = strides[i] if i < len(strides) else strides[-1]
            
            b, c, h, w = pred.shape
            # 展平特征图: [B, C, H, W] -> [B, H*W, C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

            reg_channels = 4 * self.reg_max
            all_pred_reg.append(pred[..., :reg_channels].view(bs, -1, 4, self.reg_max))
            all_pred_cls.append(pred[..., reg_channels:])

            # 生成锚点
            gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
            anchor_points = torch.stack([gx, gy], dim=-1).float() + 0.5
            all_anchors.append((anchor_points * stride).view(-1, 2))
            all_strides.append(torch.full((h * w, 1), stride, device=device))

        # 合并多尺度结果
        all_pred_reg = torch.cat(all_pred_reg, dim=1)  # [B, N, 4, 16]
        all_pred_cls = torch.cat(all_pred_cls, dim=1)  # [B, N, 19]
        all_anchors = torch.cat(all_anchors, dim=0)    # [N, 2]
        all_strides = torch.cat(all_strides, dim=0)    # [N, 1]

        # 2. 【核心修复】先解码预测框，定义 pd_boxes
        proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        pd_dist = (F.softmax(all_pred_reg, dim=-1) * proj).sum(-1) # [B, N, 4]
        
        anc_points_batch = all_anchors.unsqueeze(0).repeat(bs, 1, 1) # [B, N, 2]
        strides_batch = all_strides.unsqueeze(0) # [1, N, 1]
        
        # 计算预测框 (LTRB -> XYXY)
        pd_boxes = torch.cat([
            anc_points_batch - pd_dist[..., :2] * strides_batch, 
            anc_points_batch + pd_dist[..., 2:] * strides_batch
        ], dim=-1) # [B, N, 4]

        # 3. 整理 GT 标签 (从 [M, 6] 还原到 [B, Max_Boxes, 5])
        max_boxes = 0
        for i in range(bs):
            max_boxes = max(max_boxes, (targets[:, 0] == i).sum().item())
        
        gt_labels = torch.zeros((bs, max_boxes, 1), device=device)
        gt_bboxes = torch.zeros((bs, max_boxes, 4), device=device)
        mask_gt = torch.zeros((bs, max_boxes, 1), device=device)

        for i in range(bs):
            t = targets[targets[:, 0] == i]
            if len(t) > 0:
                gt_labels[i, :len(t)] = t[:, 1:2]
                gt_bboxes[i, :len(t)] = self.xywh2xyxy(t[:, 2:6] * 640)
                mask_gt[i, :len(t)] = 1.0

        # 4. 执行 TAL 匹配 (pd_boxes 现在已定义)
        mask_pos, align_metrics = self.get_tal_mask(
            all_pred_cls.detach().sigmoid(), 
            pd_boxes.detach(), 
            anc_points_batch, 
            gt_labels, gt_bboxes, mask_gt
        )

        # 5. 计算损失
        target_bboxes = torch.zeros_like(pd_boxes)
        target_scores = torch.zeros_like(all_pred_cls)
        fg_mask = mask_pos.any(1) 
        num_pos = fg_mask.sum().item()
        
        # 调试：每 epoch 打印一次 TAL 分配情况
        if not hasattr(self, '_debug_cnt'):
            self._debug_cnt = 0
        self._debug_cnt += 1
        if self._debug_cnt % 307 == 1:  # 约每个 epoch 打印一次
            total_anchors = fg_mask.shape[0]
            print(f"🔧 [TAL Debug] 正样本: {num_pos}/{total_anchors} "
                  f"(每图均{num_pos/max(total_anchors//16,1):.1f})")

        if fg_mask.any():
            # 获取对应的 GT 索引
            gt_idx = mask_pos.float().argmax(1) 
            b_idx, n_idx = torch.where(fg_mask) 
            selected_gt_idx = gt_idx[b_idx, n_idx]
            
            target_bboxes[b_idx, n_idx] = gt_bboxes[b_idx, selected_gt_idx]
            t_labels = gt_labels[b_idx, selected_gt_idx].squeeze(-1).long()
            
            # 计算 IoU
            pd_boxes_fg = pd_boxes[fg_mask]
            tgt_boxes_fg = target_bboxes[fg_mask]
            iou_fg = self.bbox_iou(pd_boxes_fg, tgt_boxes_fg, CIoU=False).detach().clamp(0)

            # --- 任务对齐标签 (Task-Alignment) ---
            # 提取正样本点的指标
            fg_align_metrics = align_metrics[b_idx, selected_gt_idx, n_idx]
            # 每个 GT 对应的最大指标（用于归一化）
            metric_max_per_gt = align_metrics.max(-1, keepdim=True)[0] 
            norm_align_metric = align_metrics / (metric_max_per_gt + 1e-9)
            
            # 核心标签：归一化指标 * IoU
            # 使用选取出的具体位置赋值
            target_scores[b_idx, n_idx, t_labels] = (norm_align_metric[b_idx, selected_gt_idx, n_idx] * iou_fg).to(target_scores.dtype)
       # 1. 分类损失 (BCE)
        # 使用 num_anchors 归一化，防止 target_sum 过小导致 loss 放大
        num_anchors = all_pred_cls.shape[1]  # 8400
        loss_cls = F.binary_cross_entropy_with_logits(all_pred_cls, target_scores, reduction='none').sum()
        loss_cls /= max(num_anchors, 1.0)

        # 2. 回归损失 (IoU) 与 DFL
        loss_bbox = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)

        if fg_mask.any():
            # 提取正样本的预测和标签
            num_pos = fg_mask.sum().clamp(min=1)
            
            # IoU Loss (使用 CIoU)
            iou = self.bbox_iou(pd_boxes[fg_mask], target_bboxes[fg_mask], CIoU=True)
            loss_bbox = ((1.0 - iou).sum() / num_pos)
            
            # DFL Loss
            s = all_strides.expand_as(anc_points_batch)[fg_mask]
            anchors_px = anc_points_batch[fg_mask]
            target_ltrb = torch.cat([
                (anchors_px - target_bboxes[fg_mask][:, :2]) / s,
                (target_bboxes[fg_mask][:, 2:] - anchors_px) / s
            ], dim=-1)
            
            target_ltrb = target_ltrb.clamp(0, self.reg_max - 1.01)
            loss_dfl = (self.dfl_loss(all_pred_reg[fg_mask], target_ltrb).sum() / num_pos)

        # 最终加权
        l_cls = loss_cls * self.cls_loss_weight
        l_bbox = loss_bbox * self.bbox_loss_weight
        l_dfl = loss_dfl * self.dfl_loss_weight

        # 调试打印 (建议保留，直到 Total Loss < 10)
        # print(f"Cls: {l_cls.item():.3f} | Box: {l_bbox.item():.3f} | DFL: {l_dfl.item():.3f}")
   
        self.debug_data = {
            'pd_boxes': pd_boxes[0].detach(),      # 只存 Batch 0
            'gt_bboxes': gt_bboxes[0].detach(), 
            'mask_pos': mask_pos[0].detach(), 
            'anc_points': anc_points_batch[0].detach()
        }

        return l_cls + l_bbox + l_dfl, l_cls, l_bbox, torch.tensor(0.0, device=device), l_dfl