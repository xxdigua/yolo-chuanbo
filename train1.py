import torch 
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
from tqdm import tqdm
import math
import csv
import time
from datetime import datetime

# 导入项目组件 (确保路径正确)
from models.yolov8 import yolov8_n
from utils.dataloader import get_dataloader
from utils.loss import YOLOv8Loss, visualize_tal_assignments
from utils.utils import postprocess, MAPCalculator
from models.head import Detect
# --- 辅助函数：绘制验证结果 ---
def draw_val_results(img_bgr, preds, class_names, save_path):
    img = img_bgr.copy()
    if preds is None or len(preds) == 0:
        cv2.imwrite(save_path, img)
        return
    
    for p in preds:
        x1, y1, x2, y2, conf, cls_id = p
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        idx = int(cls_id)
        
        name = class_names[idx] if idx < len(class_names) else f"ID:{idx}"
        label = f"{name} {conf:.2f}"
        
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 5), (x1 + t_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    cv2.imwrite(save_path, img)

def yolov8_specific_init(model):
    """
    针对 TypeError: 'Conv' object is not iterable 的修正版
    """
    prior = 0.01
    init_bias = -math.log((1 - prior) / prior)
    
    for m in model.modules():
        # 检查是否是检测头层 (通常包含 cv2 和 cv3)
        if hasattr(m, 'cv3') and isinstance(m.cv3, nn.ModuleList):
            # 1. 初始化分类头 (cv3)
            for x in m.cv3:
                # x 是一个 Sequential 或 Conv 模块，我们需要找到最后的 Conv2d
                if isinstance(x, nn.Sequential):
                    layer = x[-1]
                else:
                    layer = x
                
                if hasattr(layer, 'conv') and isinstance(layer.conv, nn.Conv2d):
                    nn.init.constant_(layer.conv.bias, init_bias)
                    nn.init.normal_(layer.conv.weight, std=0.01)
                elif isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.bias, init_bias)
                    nn.init.normal_(layer.weight, std=0.01)

            # 2. 初始化回归头 (cv2) - 防止初始框乱飞
            for x in m.cv2:
                if isinstance(x, nn.Sequential):
                    layer = x[-1]
                else:
                    layer = x
                
                if hasattr(layer, 'conv') and isinstance(layer.conv, nn.Conv2d):
                    nn.init.constant_(layer.conv.bias, 1.0)
                elif isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.bias, 1.0)
                    
    print(f"✅ 检测头偏置校准完成 (Bias: {init_bias:.2f})")

def train():
    # --- 1. 基础配置 ---
    num_classes = 6
    img_size = 640
    batch_size = 16
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_names = ['ore carrier', 'bulk cargo carrier', 'fishing boat',
                   'general cargo ship', 'container ship', 'passenger ship']
    
    # 路径配置
    data_root = r"D:\yolo\yolov8\data\VOCdevkit"
    train_img = r"D:\yolo\yolov8\data" 
    val_img   = r"D:\yolo\yolov8\data" 
    train_lab = r"D:\yolo\yolov8\data\VOCdevkit\labels"
    val_lab   = r"D:\yolo\yolov8\data\VOCdevkit\labels"
    train_txt = os.path.join(data_root, "train.txt")
    val_txt   = os.path.join(data_root, "val.txt")

    os.makedirs("debug_output", exist_ok=True)
    os.makedirs("val_output", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 创建训练日志CSV文件
    log_file = "logs/training_log.csv"
    log_fields = [
        'epoch', 'timestamp', 
        'train_loss_total', 'train_loss_cls', 'train_loss_box', 'train_loss_dfl',
        'val_mAP50', 'val_precision', 'val_recall',
        'learning_rate', 'epoch_time'
    ]
    
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
    
    print(f"📝 训练日志将保存至: {log_file}")

    # --- 2. 模型与优化器 ---
    model = yolov8_n(num_classes=num_classes).to(device)
    detect_layer_found = False
    for m in model.modules():
        if isinstance(m, Detect):
            m.bias_init()
            detect_layer_found = True
            break # 找到第一个 Detect 头就退出（YOLOv8 通常只有一个输出头）

    if not detect_layer_found:
        print("⚠️ 警告：在模型中未找到 Detect 层，偏置初始化未执行！")
    # ⚠️ 防爆策略 A: 如果之前一直炸，先注释掉初始化试试
    # yolov8_specific_init(model) 
    
    criterion = YOLOv8Loss(num_classes=num_classes).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 学习率预热配置（更长预热期防止分类爆炸）
    warmup_epochs = 5
    warmup_lr_init = 1e-6
    
    # ⚠️ 防爆策略 C: 暂时禁用 AMP (GradScaler)，改用全 FP32 训练以确保数值稳定
    # scaler = GradScaler('cuda') 

    # --- 3. 数据加载 ---
    train_loader = get_dataloader(train_img, train_lab, batch_size, img_size, augment=True, txt_file=train_txt)
    val_loader = get_dataloader(val_img, val_lab, batch_size, img_size, augment=False, txt_file=val_txt)

    # 最终核验数据
    test_imgs, test_targets = next(iter(train_loader))
    print(f"✅ 最终数据核验:")
    print(f"   图像像素范围: {test_imgs.min().item():.2f} ~ {test_imgs.max().item():.2f} (应为 0~1)")
    if test_targets.shape[0] > 0:
        print(f"   标签坐标范围: {test_targets[:, 2:].min().item():.4f} ~ {test_targets[:, 2:].max().item():.4f} (应为 0~1)")

    print(f"\n🚀 任务启动 | 设备: {device} | 模式: FP32 (防溢出增强版)")
    print("="*60)
    best_map = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # --- 学习率预热 ---
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            warmup_lr = warmup_lr_init + (1e-3 - warmup_lr_init) * warmup_factor
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        
        # --- 训练阶段 ---
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
        epoch_loss_items = torch.zeros(4, device=device) 

        for i, (imgs, labels) in pbar:
            # ⚠️ 数据已在 Dataloader 归一化过了，这里只需 move to device
            imgs = imgs.to(device)
            targets = labels.to(device) 
            
            optimizer.zero_grad()
            
            # 禁用 autocast，改用标准 FP32
            outputs = model(imgs)
            loss, l_cls, l_box, _, l_dfl = criterion(outputs, targets)

            if torch.isnan(loss):
                print(f"❌ 警告：检测到 NaN 损失！跳过此 Batch。")
                continue

            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
         
            # 统计
            loss_items = torch.stack([loss, l_cls, l_box, l_dfl]).detach()
            epoch_loss_items += loss_items
            # ... 在 loss.backward() 之前 ...
            if i % 50 == 0:
                # 采样第一个样本的分类输出均值（Sigmoid 之后）
                with torch.no_grad():
                    cls_prob = torch.sigmoid(outputs[0][0, 64:, :]).mean().item()
                    # 理想情况下，这个值在第一轮应该 < 0.1
                    if cls_prob > 0.5:
                        print(f"\n⚠️ 警告：分类层输出过高 ({cls_prob:.2f})，损失可能爆炸！")
            pbar.set_postfix({
                'Total': f"{loss_items[0]:.3f}",
                'Cls': f"{loss_items[1]:.3f}",
                'Box': f"{loss_items[2]:.3f}",
                'DFL': f"{loss_items[3]:.3f}"
            })

            # 训练可视化 (每 200 step)
            if i % 200 == 0:
                with torch.no_grad():
                    d = criterion.debug_data
                    # 转换回 BGR 用于 OpenCV
                    vis_img = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    visualize_tal_assignments(vis_img, d['pd_boxes'], d['gt_bboxes'], 
                                             d['mask_pos'], d['anc_points'], 
                                             save_path=f"debug_output/E{epoch+1}_B{i}_train.jpg")

        # --- 每个 Epoch 结束打印总结 ---
        avg_loss = epoch_loss_items / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n✨ Epoch {epoch+1} 总结:")
        print(f"   平均损失 -> 总计: {avg_loss[0]:.4f} | 分类: {avg_loss[1]:.4f} | 回归: {avg_loss[2]:.4f} | DFL: {avg_loss[3]:.4f}")
        print(f"   当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   耗时: {epoch_time:.1f}秒")

        scheduler.step()

        # 准备日志数据
        log_data = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss_total': avg_loss[0].item(),
            'train_loss_cls': avg_loss[1].item(),
            'train_loss_box': avg_loss[2].item(),
            'train_loss_dfl': avg_loss[3].item(),
            'val_mAP50': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }

        # --- 验证阶段 ---
        if (epoch + 1) % 5 == 0 or epoch == 0: 
            model.eval()
            print(f"🔍 正在执行 Epoch {epoch+1} 效验...")
            metric_logger = MAPCalculator(iou_threshold=0.5)
            
            with torch.no_grad():
                for v_idx, (v_imgs, v_labels) in enumerate(val_loader):
                    v_imgs = v_imgs.to(device)
                    v_outputs = model(v_imgs)
                    
                    preds_raw = v_outputs[0] if isinstance(v_outputs, (list, tuple)) else v_outputs
                    
                    # 原始信号采样打印 (只打印第一个 Batch 的前 2 个框)
                    if v_idx == 0:
                        raw_sample = preds_raw[0]  # [8400, 4+nc]
                        scores = torch.sigmoid(raw_sample[:, 4:])
                        boxes = raw_sample[:, :4]
                        
                        # 详细统计
                        print(f"🧪 [E{epoch+1} 预测采样]")
                        print(f"   Score: max={scores.max():.3f}, mean={scores.mean():.3f}, >0.5比例={(scores>0.5).float().mean():.2%}")
                        print(f"   Box x1: min={boxes[:,0].min():.1f}, max={boxes[:,0].max():.1f}")
                        print(f"   Box y1: min={boxes[:,1].min():.1f}, max={boxes[:,1].max():.1f}")
                        print(f"   Box x2: min={boxes[:,2].min():.1f}, max={boxes[:,2].max():.1f}")
                        print(f"   Box y2: min={boxes[:,3].min():.1f}, max={boxes[:,3].max():.1f}")
                    
                    preds = postprocess(preds_raw, conf_thres=0.6, iou_thres=0.6, img_size=img_size)
                    
                    if v_idx == 0:
                        # 调试：统计有效框数量
                        total_preds = sum(len(p) for p in preds)
                        print(f"🔍 [后处理] 有效预测框总数: {total_preds}")
                    
                    metric_logger.update(preds, v_labels, img_size=img_size)
                    
                    if v_idx == 0:
                        vis_img = (v_imgs[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        val_vis = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR) 
                        vis_preds_draw = preds[0][preds[0][:, 4] > 0.1] if len(preds[0]) > 0 else []
                        draw_val_results(val_vis, vis_preds_draw, class_names, f"val_output/epoch_{epoch+1}_val.jpg")

            stats = metric_logger.compute()
            print(f"📊 验证结果 -> mAP50: {stats['mAP50']:.4f} | Precision: {stats['precision']:.4f} | Recall: {stats['recall']:.4f}")
            
            # 更新日志数据
            log_data['val_mAP50'] = stats['mAP50']
            log_data['val_precision'] = stats['precision']
            log_data['val_recall'] = stats['recall']
            
            if stats['mAP50'] > best_map:
                best_map = stats['mAP50']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mAP': best_map}, "weights/best.pth")
                print(f"⭐ 已更新 best.pth")
        
        # 保存日志到CSV
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(log_data)
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, "weights/last.pth")
        print("-" * 60)

if __name__ == "__main__":
    train()