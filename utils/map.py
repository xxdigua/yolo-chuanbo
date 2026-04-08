import numpy as np
import torch

class MAPCalculator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.stats = []  # 存储 (tp, conf, pred_cls, target_cls)

    def update(self, preds, targets, img_size=640):
        """
        preds: list of np.array [x1, y1, x2, y2, conf, cls] (来自 postprocess)
        targets: batch of labels [batch_idx, cls, cx, cy, w, h] (归一化)
        """
        for i, pred in enumerate(preds):
            # 筛选当前图片的标注
            # 假设 targets 是 [N, 6] 格式：[batch_id, cls, cx, cy, w, h]
            curr_target = targets[targets[:, 0] == i]
            
            if len(pred) == 0:
                if len(curr_target) > 0:
                    # 有GT但没预测出：不添加 TP，仅记录 GT 类别
                    self.stats.append((np.zeros(0), np.zeros(0), np.zeros(0), curr_target[:, 1]))
                continue

            # 转换 GT 格式为 [x1, y1, x2, y2]
            gt_boxes = curr_target[:, 2:].copy()
            gt_boxes[:, [0, 2]] *= img_size # cx, w
            gt_boxes[:, [1, 3]] *= img_size # cy, h
            
            # cx,cy,w,h -> x1,y1,x2,y2
            gt_x1y1 = gt_boxes[:, :2] - gt_boxes[:, 2:] / 2
            gt_x2y2 = gt_boxes[:, :2] + gt_boxes[:, 2:] / 2
            gt_boxes = np.concatenate([gt_x1y1, gt_x2y2], axis=1)
            gt_labels = curr_target[:, 1]

            # 匹配逻辑
            tp = np.zeros(len(pred))
            conf = pred[:, 4]
            pred_cls = pred[:, 5]
            
            if len(gt_boxes) > 0:
                matched_gt = []
                # 按置信度排序预测框
                sort_idx = np.argsort(-conf)
                for p_idx in sort_idx:
                    p_box = pred[p_idx, :4]
                    p_c = pred[p_idx, 5]
                    
                    # 计算与所有同类 GT 的 IoU
                    ious = self._box_iou(p_box, gt_boxes)
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for g_idx, g_box in enumerate(gt_boxes):
                        if gt_labels[g_idx] == p_c and g_idx not in matched_gt:
                            if ious[g_idx] > best_iou:
                                best_iou = ious[g_idx]
                                best_gt_idx = g_idx
                    
                    if best_iou >= self.iou_threshold:
                        tp[p_idx] = 1
                        matched_gt.append(best_gt_idx)

            self.stats.append((tp, conf, pred_cls, gt_labels))

    def compute(self):
        """计算最终的 mAP@50, Precision, Recall"""
        if not self.stats: 
            return 0.0, 0.0, 0.0
        
        # 合并所有图片的结果
        # tp: 是否匹配成功, conf: 置信度, pred_cls: 预测类别, target_cls: 真实类别
        tp = np.concatenate([x[0] for x in self.stats])
        conf = np.concatenate([x[1] for x in self.stats])
        pred_cls = np.concatenate([x[2] for x in self.stats])
        target_cls = np.concatenate([x[3] for x in self.stats])
        
        unique_classes = np.unique(target_cls)
        ap_list = []
        p_list = []
        r_list = []
        
        for c in unique_classes:
            c_mask = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_pred = c_mask.sum()
            
            if n_pred == 0 or n_gt == 0:
                ap_list.append(0)
                p_list.append(0)
                r_list.append(0)
                continue
            
            # 按置信度降序排列
            sort_idx = np.argsort(-conf[c_mask])
            c_tp = tp[c_mask][sort_idx]
            c_fp = 1 - c_tp
            
            # 累积计算
            f_tp = np.cumsum(c_tp)
            f_fp = np.cumsum(c_fp)
            
            recall_curve = f_tp / (n_gt + 1e-16)
            precision_curve = f_tp / (f_tp + f_fp + 1e-16)
            
            # 计算该类别的 AP
            ap = self._compute_ap(recall_curve, precision_curve)
            ap_list.append(ap)
            
            # 提取该类别的 Precision 和 Recall (通常取 F1-score 最大点的值)
            f1 = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-16)
            best_idx = np.argmax(f1)
            p_list.append(precision_curve[best_idx])
            r_list.append(recall_curve[best_idx])
            
        # 返回平均值
        return {
            'mAP50': np.mean(ap_list),
            'precision': np.mean(p_list),
            'recall': np.mean(r_list)
        }

    def _compute_ap(self, recall, precision):
        """全点插值计算 PR 曲线面积"""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        # 让 Precision 变得单调递减 (包络线)
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        # 计算积分
        return np.trapz(mpre, mrec)

    def _box_iou(self, box1, boxes):
        """计算一个框与多个框的 IoU"""
        inter_x1 = np.maximum(box1[0], boxes[:, 0])
        inter_y1 = np.maximum(box1[1], boxes[:, 1])
        inter_x2 = np.minimum(box1[2], boxes[:, 2])
        inter_y2 = np.minimum(box1[3], boxes[:, 3])
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter_area / (area1 + area2 - inter_area + 1e-9)