import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageEnhance

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=False, txt_file=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        
        # 加载图片列表
        if txt_file and os.path.exists(txt_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                self.img_files = [line.strip() for line in f if line.strip()]
        else:
            self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            img_file = self.img_files[idx].strip().replace('/', os.sep).replace('\\', os.sep)
            img_path = img_file if os.path.isabs(img_file) else os.path.join(self.img_dir, img_file)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image: {img_path}")

            # 1. 加载图像
            img = Image.open(img_path).convert('RGB')
            
            # 2. 加载标签 [class, x, y, w, h] (均为归一化值)
            img_basename = os.path.basename(img_file)
            label_name = os.path.splitext(img_basename)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_name)
            
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) == 5:
                            # 简单的数值裁剪，防止原始标签微小越界
                            parts[1:] = [np.clip(p, 0.0, 1.0) for p in parts[1:]]
                            labels.append(parts)
            
            labels = np.array(labels) if len(labels) > 0 else np.zeros((0, 5))

            # 3. 数据增强 (Augmentation)
            if self.augment:
                # A. 颜色增强 (亮度、对比度)
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.8, 1.2))
                
                # B. 水平翻转 (注意：翻转图片时，x 坐标也要翻转)
                if random.random() > 0.5:
                    img = ImageOps.mirror(img)
                    if len(labels) > 0:
                        # YOLO 格式下，翻转后的 x_center = 1.0 - x_center
                        labels[:, 1] = 1.0 - labels[:, 1]

            # 4. Resize 到模型输入尺寸
            img = img.resize((self.img_size, self.img_size))
            
            # 5. 转换为 Tensor [C, H, W] 并归一化到 [0, 1]
            img_data = np.array(img).transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img_data).float() / 255.0
            
            # 6. 最终过滤无效的小框
            valid_labels = []
            for label in labels:
                if label[3] > 0.001 and label[4] > 0.001: # w, h 必须大于 0.1% 图像大小
                    valid_labels.append(label)
            
            if len(valid_labels) > 0:
    # 先转为 numpy 矩阵，再转为 tensor，速度提升明显
                target = torch.from_numpy(np.array(valid_labels, dtype=np.float32))
            else:
                target = torch.zeros((0, 5), dtype=torch.float32)
            
            return img_tensor, target

        except Exception as e:
            print(f"⚠️ [Data Error] Index {idx} failed: {e}. Trying next...")
            return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch):
    """
    将一个 batch 的数据整理成模型需要的格式
    返回: 
        imgs: [Batch, 3, 640, 640]
        targets: [Total_Objects, 6] -> [batch_idx, class, x, y, w, h]
    """
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs)
    
    new_targets = []
    for i, tgt in enumerate(targets):
        if tgt.shape[0] > 0:
            num_obj = tgt.shape[0]
            batch_idx = torch.full((num_obj, 1), i, dtype=torch.float32)
            item = torch.cat([batch_idx, tgt], dim=1)
            new_targets.append(item)
    
    targets = torch.cat(new_targets, dim=0) if len(new_targets) > 0 else torch.zeros((0, 6))
    return imgs, targets

def get_dataloader(img_dir, label_dir, batch_size=8, img_size=640, augment=False, num_workers=4, txt_file=None):
    dataset = YOLODataset(img_dir, label_dir, img_size, augment, txt_file)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, # 训练集通常需要 shuffle
        num_workers=num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn
    )