import torch

class TrainHealthBook:
    def __init__(self, num_classes=19):
        self.num_classes = num_classes
        self.alert_count = 0

    def check(self, loss_items, targets, pred_cls):
        """
        loss_items: [total_loss, cls_loss, box_loss, dfl_loss]
        targets: build_targets 后的 [M, 6]
        pred_cls: 模型输出的分类原始值 (logits)
        """
        status = "✅ 正常"
        alerts = []

        # 1. 检测分类损失是否过大 (Focal Loss 下正常应 < 5.0)
        if loss_items[1] > 20.0:
            alerts.append(f"🚩 分类损失异常 ({loss_items[1]:.2f})：检查背景噪声或标签是否越界")

        # 2. 检测正样本分配 (TAL 是否抓到了目标)
        num_targets = targets.shape[0]
        if num_targets == 0:
            alerts.append("⚠️ 警告：当前 Batch 无有效标签，跳过训练")
        
        # 3. 检测梯度爆炸风险 (Logits 是否过大)
        max_logit = pred_cls.abs().max().item()
        if max_logit > 50.0:
            alerts.append(f"🔥 梯度爆炸风险：Logits 最大值达 {max_logit:.1f}，建议调低学习率")

        # 4. 检测类别索引合法性
        if num_targets > 0:
            target_cls = targets[:, 1]
            if (target_cls >= self.num_classes).any() or (target_cls < 0).any():
                alerts.append(f"❌ 标签错误：发现类别索引超出 0-{self.num_classes-1} 范围！")

        if alerts:
            status = "❌ 异常"
            for a in alerts: print(a)
        
        return status