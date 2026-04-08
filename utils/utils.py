import torch
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F
# ==========================================
# 1. 基础坐标转换与 IoU 工具
# ==========================================

def postprocess(predictions, conf_thres=0.25, iou_thres=0.45, img_size=640):
    """
    修正版后处理：强制 LTRB 正向化 + 边界裁剪
    """
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0] 
    
    batch_results = []
    for pred in predictions:
        bboxes = pred[:, :4]  # [8400, 4]
        scores = pred[:, 4:].sigmoid()   # [8400, nc]
        
        conf, cls_idx = scores.max(dim=1, keepdim=True)
        mask = conf.view(-1) > conf_thres
        
        if not mask.any():
            batch_results.append(np.empty((0, 6)))
            continue
            
        v_boxes = bboxes[mask]
        v_conf = conf[mask]
        v_cls = cls_idx[mask].float()

        # 🛑 核心修正：强行将 [raw_x1, raw_y1, raw_x2, raw_y2] 转换为正向矩形
        x_raw = v_boxes[:, [0, 2]]
        y_raw = v_boxes[:, [1, 3]]
        
        # ... 前面代码保持不变 ...
        x1 = x_raw.min(1)[0].clamp(0, img_size)
        y1 = y_raw.min(1)[0].clamp(0, img_size)
        x2 = x_raw.max(1)[0].clamp(0, img_size)
        y2 = y_raw.max(1)[0].clamp(0, img_size)
        
        # 过滤掉太小的框（放宽阈值以保留更多有效预测）
        w = x2 - x1
        h = y2 - y1
        valid = (w > 2.0) & (h > 2.0) # 放宽到 2 像素
        if not valid.any():
            batch_results.append(np.empty((0, 6)))
            continue

        final_boxes = torch.stack([x1, y1, x2, y2], dim=1)[valid]
        final_conf = v_conf[valid]
        final_cls = v_cls[valid]

        combined = torch.cat([final_boxes, final_conf, final_cls], dim=1)
        
        # 执行 NMS (需确保你已定义 nms 函数)
        batch_results.append(nms(combined, conf_thres, iou_thres))
        
    return batch_results

def nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    if prediction.shape[0] == 0:
        return np.empty((0, 6))

    # 1. 按照置信度排序，只取前 30000 个，加速处理
    prediction = prediction[prediction[:, 4].argsort(descending=True)[:30000]]

    boxes = prediction[:, :4]
    scores = prediction[:, 4]
    classes = prediction[:, 5]

    # 类别偏移多类别 NMS
    offsets = classes * 4096
    boxes_for_nms = boxes + offsets.view(-1, 1)

    keep = torchvision.ops.nms(boxes_for_nms, scores, iou_thres)
    
    # 限制最大检测数
    keep = keep[:max_det]
    
    return prediction[keep].cpu().numpy()

def bbox_iou(box1, box2, eps=1e-9):
    """计算 IoU"""
    x1 = torch.max(box1[0], box2[:, 0])
    y1 = torch.max(box1[1], box2[:, 1])
    x2 = torch.min(box1[2], box2[:, 2])
    y2 = torch.min(box1[3], box2[:, 3])
    
    w = (x2 - x1).clamp(0)
    h = (y2 - y1).clamp(0)
    inter = w * h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter + eps
    return inter / union

# ==========================================
# 3. 评估类 (补全了核心数学逻辑)
# ==========================================

class MAPCalculator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.stats = []

    def update(self, preds, targets, img_size=640):
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        for i, pred in enumerate(preds):
            # 确保 pred 是 numpy 格式且不为空
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            
            curr_target = targets[targets[:, 0] == i]
            
            if len(curr_target) == 0:
                if len(pred) > 0:
                    self.stats.append((np.zeros(len(pred)), pred[:, 4], pred[:, 5], np.zeros(0)))
                continue

            gt_labels = curr_target[:, 1]
            gt_boxes = curr_target[:, 2:6].copy() # 显式取 2:6，即 [cx, cy, w, h]
            
            # 🛑 关键检查：如果 gt_boxes 是归一化的，强制放大
            # 增加一个逻辑：如果平均值很小，通常也是归一化的
            if gt_boxes[:, 2:].max() <= 1.01: 
                gt_boxes *= img_size
            
            # 转换 GT 格式为 [x1, y1, x2, y2]
            gt_px = np.zeros_like(gt_boxes)
            gt_px[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
            gt_px[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
            gt_px[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
            gt_px[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
            gt_px = np.clip(gt_px, 0, img_size)

            # ... 剩下的 TP/FP 匹配逻辑保持不变 ...
            # (确保你使用了之前提供的 _box_iou_np 进行计算)

            # 3. 初始化当前图的 True Positive 向量
            num_pred = len(pred)
            tp = np.zeros(num_pred)
            
            if num_pred > 0:
                # 按照置信度从高到低排序（mAP 计算标准要求）
                sort_idx = np.argsort(-pred[:, 4])
                pred = pred[sort_idx]
                
                matched_gt = set() # 记录已经被匹配过的 GT 索引
                
                for p_idx in range(num_pred):
                    p_box = pred[p_idx, :4]
                    p_cls = pred[p_idx, 5]
                    
                    # 计算当前预测框与所有 GT 的 IoU
                    ious = self._box_iou_np(p_box, gt_px)
                    
                    # 寻找 IoU 最大且类别匹配、未被占用的 GT
                    best_iou = -1
                    best_gt_idx = -1
                    
                    for g_idx in range(len(gt_px)):
                        if gt_labels[g_idx] == p_cls and g_idx not in matched_gt:
                            if ious[g_idx] > best_iou:
                                best_iou = ious[g_idx]
                                best_gt_idx = g_idx
                    
                    # 判定是否为 TP
                    if best_iou >= self.iou_threshold:
                        tp[p_idx] = 1
                        matched_gt.add(best_gt_idx) # 一个 GT 只能被匹配一次

                # 将结果存入 stats (注意顺序要和排序后的 pred 对应)
                self.stats.append((tp, pred[:, 4], pred[:, 5], gt_labels))

    def _box_iou_np(self, b1, b2s):
        """计算一个框与一组框的 IoU (Numpy 版)"""
        inter_x1 = np.maximum(b1[0], b2s[:, 0])
        inter_y1 = np.maximum(b1[1], b2s[:, 1])
        inter_x2 = np.minimum(b1[2], b2s[:, 2])
        inter_y2 = np.minimum(b1[3], b2s[:, 3])
        
        inter_w = (inter_x2 - inter_x1).clip(0)
        inter_h = (inter_y2 - inter_y1).clip(0)
        inter_area = inter_w * inter_h
        
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        b2_areas = (b2s[:, 2] - b2s[:, 0]) * (b2s[:, 3] - b2s[:, 1])
        
        union_area = b1_area + b2_areas - inter_area + 1e-9
        return inter_area / union_area

    def compute(self):
        if not self.stats: 
            return {"mAP50": 0, "precision": 0, "recall": 0}
        
        tp = np.concatenate([x[0] for x in self.stats])
        conf = np.concatenate([x[1] for x in self.stats])
        pred_cls = np.concatenate([x[2] for x in self.stats])
        target_cls = np.concatenate([x[3] for x in self.stats])
        
        unique_classes = np.unique(target_cls)
        ap_all, p_all, r_all = [], [], []
        
        for c in unique_classes:
            i = pred_cls == c
            # 🛑 这里的 target_cls 是所有图片的 GT 集合
            # 确保统计的是所有图片中该类别的 GT 总数
            n_gt = (target_cls == c).sum() 
            
            if n_gt == 0: continue # 如果验证集没这个类，跳过
            
            # ... 排序 ...
            fpc = (1 - tp[i][np.argsort(-conf[i])]).cumsum()
            tpc = (tp[i][np.argsort(-conf[i])]).cumsum()
            
            recall = tpc / n_gt
            precision = tpc / (tpc + fpc)
            
            # 🛑 AP 计算：增加边界处理，防止 recall 不从 0 开始
            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0]))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            ap_all.append(np.trapz(mpre, mrec))
            
            # 防止空数组访问
            if len(precision) > 0:
                p_all.append(precision[-1])
            else:
                p_all.append(0)
            
            if len(recall) > 0:
                r_all.append(recall[-1])
            else:
                r_all.append(0)
            
        return {
            "mAP50": np.mean(ap_all) if ap_all else 0,
            "precision": np.mean(p_all) if p_all else 0,
            "recall": np.mean(r_all) if r_all else 0
        }

# ==========================================
# 4. 可视化工具
# ==========================================

def draw_val_results(img, preds, class_names, save_path):
    """
    img: 已经是 BGR 格式的 numpy 数组 (0-255)
    preds: [N, 6] 的预测框 (x1, y1, x2, y2, conf, cls)
    """
    # 这一步很重要，防止修改原始内存图像
    draw_img = img.copy()
    
    if preds is not None and len(preds) > 0:
        for p in preds:
            x1, y1, x2, y2, conf, cls_id = p
            # 转换为整数坐标
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 画矩形框
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 写标签
            idx = int(cls_id)
            name = class_names[idx] if idx < len(class_names) else f"ID:{idx}"
            label = f"{name} {conf:.2f}"
            cv2.putText(draw_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 最终保存
    cv2.imwrite(save_path, draw_img)
    # print(f"✅ 验证图已保存: {save_path}")