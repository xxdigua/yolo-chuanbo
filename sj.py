import torch
import torch.nn.functional as F

def debug_yolov8_decoding(img_size=640, num_classes=6):
    # 1. 模拟模型原始输出 [Batch, 70, 8400] 
    # 70 = 64(DFL) + 6(类别)
    batch_size = 1
    num_anchors = 8400
    reg_max = 16
    raw_output = torch.randn(batch_size, 64 + num_classes, num_anchors)
    
    print(f"📦 原始输出形状: {raw_output.shape}")

    # 2. 维度转换 [B, 70, 8400] -> [B, 8400, 70]
    preds = raw_output.transpose(1, 2)
    pred = preds[0] # 取第一张图 [8400, 70]

    # 3. 拆分 DFL 坐标和类别分数
    bbox_raw = pred[:, :64]      # [8400, 64]
    cls_logits = pred[:, 64:]    # [8400, 6]
    
    # 4. 解码 DFL (最核心步骤)
    # 将 64 维转为 [8400, 4, 16]，4代表 left, top, right, bottom
    bbox_split = bbox_raw.view(-1, 4, reg_max)
    # 对最后一位做 Softmax，让分布权和为 1
    bbox_softmax = F.softmax(bbox_split, dim=-1)
    # 乘以系数矩阵 [0, 1, 2, ..., 15] 得到期望偏移量
    coeffs = torch.arange(reg_max).float()
    bbox_dist = (bbox_softmax * coeffs).sum(-1) # 得到 [8400, 4]
    
    print(f"📍 解码后的偏移量 (Distance) 示例 (前5个点):\n{bbox_dist[:5]}")

    # 5. 生成网格 (Grids) 和 步长 (Strides)
    # 这是将“偏移量”转为“像素坐标”的基准
    grids = []
    strides = []
    for s in [8, 16, 32]:
        nx = ny = img_size // s
        y, x = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
        # +0.5 是为了移动到网格中心
        grid = torch.stack((x, y), 2).view(-1, 2) + 0.5
        grids.append(grid)
        strides.append(torch.full((ny * nx, 1), s))
    
    all_grids = torch.cat(grids, 0)
    all_strides = torch.cat(strides, 0)

    # 6. 计算最终像素坐标 (LTRB 格式)
    x1 = (all_grids[:, 0:1] - bbox_dist[:, 0:1]) * all_strides
    y1 = (all_grids[:, 1:2] - bbox_dist[:, 1:2]) * all_strides
    x2 = (all_grids[:, 0:1] + bbox_dist[:, 2:3]) * all_strides
    y2 = (all_grids[:, 1:2] + box_dist[:, 3:4] if 'box_dist' in locals() else bbox_dist[:, 3:4]) * all_strides
    
    final_bboxes = torch.cat([x1, y1, x2, y2], dim=1)

    print("-" * 30)
    print(f"✅ 解码完成！")
    print(f"网格 (Grids) 形状: {all_grids.shape}")
    print(f"像素坐标 (Bboxes) 形状: {final_bboxes.shape}")
    print(f"第一个锚点的最终坐标 [x1, y1, x2, y2]:\n{final_bboxes[0].detach().numpy()}")

if __name__ == "__main__":
    debug_yolov8_decoding()