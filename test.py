import torch
import cv2
import os
import numpy as np
import time
from pathlib import Path

from models.yolov8 import yolov8_n
from utils.utils import postprocess


def preprocess(img, img_size=640):
    """
    图像预处理：直接 resize 到 img_size × img_size（与训练一致）
    返回:
        input_tensor: [1, 3, H, W] 归一化后的tensor
        resized_img: resize 后的 BGR 图像（用于绘制预测框）
    """
    h, w = img.shape[:2]

    # 直接 resize 到 img_size × img_size（与训练一致，不保持长宽比）
    resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB, 归一化到 [0, 1], 转换为 tensor [C, H, W]
    rgb_img = resized[:, :, ::-1].astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0)

    return input_tensor, resized


def plot_bboxes(img, bboxes, class_names=None, line_thickness=2):
    """
    在图像上绘制检测框（与验证时一致，直接在 resize 后的图像上绘制）
    参数:
        img: BGR图像（640x640）
        bboxes: [N, 6] 检测结果 (x1, y1, x2, y2, conf, cls) - 640x640 尺度
        class_names: 类别名称列表
        line_thickness: 线条粗细
    返回:
        绘制后的图像
    """
    draw_img = img.copy()

    # 定义颜色列表 (BGR格式)
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255),
        (128, 255, 0), (255, 0, 128), (0, 255, 128),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (192, 192, 192)
    ]

    for i, box in enumerate(bboxes):
        x1, y1, x2, y2, conf, cls_id = box

        # 直接使用预测坐标（640x640 尺度），不需要还原
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)

        # 选择颜色
        color = colors[cls_id % len(colors)]

        # 绘制矩形框
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, line_thickness)

        # 准备标签文本
        if class_names and cls_id < len(class_names):
            label = f"{class_names[cls_id]} {conf:.2f}"
        else:
            label = f"Class_{cls_id} {conf:.2f}"

        # 计算文本大小和位置
        font_scale = 0.6
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # 绘制背景填充
        y1_text = max(y1 - text_h - baseline - 5, 0)
        cv2.rectangle(
            draw_img,
            (x1, y1_text),
            (x1 + text_w, y1_text + text_h + baseline + 5),
            color,
            -1
        )

        # 绘制文本
        cv2.putText(
            draw_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return draw_img


def test_single_image(model, image_path, save_path, device,
                      conf_thres=0.50, iou_thres=0.45,
                      class_names=None, img_size=640):
    """
    测试单张图片
    """
    print(f"\n{'='*60}")
    print(f"📷 测试图片: {image_path}")
    print(f"{'='*60}")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None

    original_h, original_w = img.shape[:2]
    print(f"📐 原图尺寸: {original_w} x {original_h}")

    # 预处理
    start_preprocess = time.time()
    input_tensor, resized_img = preprocess(img, img_size=img_size)
    preprocess_time = time.time() - start_preprocess
    print(f"⏱️ 预处理耗时: {preprocess_time*1000:.2f} ms")

    input_tensor = input_tensor.to(device)

    # 推理
    start_inference = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
    inference_time = time.time() - start_inference
    print(f"⏱️ 推理耗时: {inference_time*1000:.2f} ms")

    # 后处理
    start_postprocess = time.time()

    # 处理输出格式
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

    print(f"📊 输出形状: {outputs.shape}")

    # 执行后处理
    bboxes = postprocess(outputs, conf_thres=conf_thres, iou_thres=iou_thres, img_size=img_size)
    postprocess_time = time.time() - start_postprocess
    print(f"⏱️ 后处理耗时: {postprocess_time*1000:.2f} ms")
    print(f"⏱️ 总耗时: {(preprocess_time+inference_time+postprocess_time)*1000:.2f} ms")

    # 显示检测结果
    if len(bboxes[0]) > 0:
        print(f"\n✅ 检测到 {len(bboxes[0])} 个目标:")
        print("-" * 60)
        
        for i, box in enumerate(bboxes[0]):
            x1, y1, x2, y2, conf, cls_id = box
            cls_name = class_names[int(cls_id)] if class_names and int(cls_id) < len(class_names) else f"Class_{int(cls_id)}"
            print(f"  [{i+1}] {cls_name:<20} | 置信度: {conf:.3f} | "
                  f"位置: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f}) | "
                  f"尺寸: ({x2-x1:.0f} x {y2-y1:.0f})")
        print("-" * 60)

        # 绘制结果（在 resize 后的 640x640 图像上绘制）
        result_img = plot_bboxes(resized_img, bboxes[0], class_names)

        # 保存结果
        cv2.imwrite(save_path, result_img)
        print(f"💾 结果已保存至: {save_path}")

        return {
            'image_path': image_path,
            'num_detections': len(bboxes[0]),
            'detections': bboxes[0],
            'time': {
                'preprocess': preprocess_time,
                'inference': inference_time,
                'postprocess': postprocess_time,
                'total': preprocess_time + inference_time + postprocess_time
            }
        }
    else:
        print(f"\n⚠️ 未检测到任何目标 (尝试降低 conf_thres 当前值: {conf_thres})")

        # 保存 resize 后的图像
        cv2.imwrite(save_path, resized_img)
        print(f"💾 图像已保存至: {save_path}")

        return {
            'image_path': image_path,
            'num_detections': 0,
            'detections': [],
            'time': {
                'preprocess': preprocess_time,
                'inference': inference_time,
                'postprocess': postprocess_time,
                'total': preprocess_time + inference_time + postprocess_time
            }
        }


def test_batch_images(model, image_dir, output_dir, device,
                      conf_thres=0.50, iou_thres=0.45,
                      class_names=None, img_size=640,
                      max_images=None, image_list_file=None):
    """
    批量测试多张图片
    
    参数:
        image_list_file: 图片列表文件路径(如test.txt)，每行一个相对路径
                        如果提供此参数，将优先使用文件中的图片列表
    """
    print(f"\n{'='*60}")
    print(f"📁 开始批量测试")
    print(f"📂 图片目录: {image_dir}")
    print(f"📂 输出目录: {output_dir}")
    if image_list_file:
        print(f"📋 图片列表: {image_list_file}")
    print(f"{'='*60}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取图片文件列表
    image_files = []
    
    if image_list_file and os.path.exists(image_list_file):
        # 从txt文件读取图片列表
        print(f"📖 从文件读取图片列表...")
        with open(image_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 构建完整路径
                    if os.path.isabs(line):
                        img_path = line
                    else:
                        img_path = os.path.join(image_dir, line)
                    
                    if os.path.exists(img_path):
                        image_files.append(Path(img_path))
                    else:
                        print(f"⚠️ 图片不存在: {img_path}")
        
        print(f"✅ 从文件读取到 {len(image_files)} 张图片")
    else:
        # 扫描目录下所有图片
        if image_list_file:
            print(f"⚠️ 图片列表文件不存在: {image_list_file}")
            print(f"📂 改为扫描目录: {image_dir}")
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for ext in valid_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        image_files = sorted(set(image_files))

    if not image_files:
        print(f"❌ 未找到任何图片文件")
        return []

    if max_images:
        image_files = image_files[:max_images]

    print(f"📊 共找到 {len(image_files)} 张图片待测试\n")

    results = []
    total_detections = 0
    total_time = 0

    for idx, img_path in enumerate(image_files, 1):
        # 生成输出路径
        output_path = os.path.join(output_dir, f'result_{img_path.name}')

        # 测试单张图片
        result = test_single_image(
            model=model,
            image_path=str(img_path),
            save_path=output_path,
            device=device,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            class_names=class_names,
            img_size=img_size
        )

        if result:
            results.append(result)
            total_detections += result['num_detections']
            total_time += result['time']['total']

        # 每10张打印一次进度
        if idx % 10 == 0 or idx == len(image_files):
            print(f"\n📈 进度: [{idx}/{len(image_files)}] | "
                  f"已检测目标总数: {total_detections} | "
                  f"平均耗时: {total_time/idx*1000:.2f} ms/张\n")

    # 打印汇总统计
    print(f"\n{'='*60}")
    print(f"📊 批量测试完成!")
    print(f"{'='*60}")
    print(f"📷 总图片数: {len(results)}")
    print(f"🎯 总检测数: {total_detections}")
    print(f"⏱️ 总耗时: {total_time:.2f} s")
    print(f"⏱️ 平均耗时: {total_time/max(len(results),1)*1000:.2f} ms/张")
    print(f"⏱️ FPS: {len(results)/max(total_time,0.001):.2f}")
    print(f"💾 结果保存在: {output_dir}")
    print(f"{'='*60}\n")

    return results


# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    # ==================== 配置区域 ====================
    
    # 模型配置
    WEIGHT_PATH = "D:/yolo/yolov8/weights/last.pth"
    NUM_CLASSES = 6  # 根据你的训练配置修改
    
    # 类别名称 (根据你的数据集修改)
    CLASS_NAMES = [
        'ore carrier', 'bulk cargo carrier', 'fishing boat',
        'general cargo ship', 'container ship', 'passenger ship'
    ]
    
    # 推理参数
    IMG_SIZE = 640
    CONF_THRES = 0.50  # 置信度阈值 (可调整)
    IOU_THRES = 0.45   # NMS IoU阈值 (可调整)
    
    # 测试模式选择: "single" 或 "batch"
    TEST_MODE = "batch"
    
    # 单张图片测试路径
    SINGLE_IMAGE_PATH = r"D:\yolo\yolov8\data\VOCdevkit\images\001617.jpg"
    SINGLE_SAVE_PATH = "D:/yolo/yolov8/data/result.jpg"
    
    # 批量测试路径
    BATCH_IMAGE_DIR = r"D:\yolo\yolov8\data"
    test_txt = os.path.join(BATCH_IMAGE_DIR, r"VOCdevkit\test.txt")
    BATCH_OUTPUT_DIR = "D:/yolo/yolov8/data/test_results"
    MAX_BATCH_IMAGES = None  # 设置为None表示测试所有图片，或指定数字如100
    
    # ==================================================
    
    print("\n" + "="*60)
    print("🚀 YOLOv8 目标检测测试系统")
    print("="*60 + "\n")
    
    # 1. 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 2. 加载模型
    print(f"\n🔧 正在加载模型...")
    model = yolov8_n(num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists(WEIGHT_PATH):
        try:
            checkpoint = torch.load(WEIGHT_PATH, map_location=device)
            
            # 兼容不同的权重格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 权重加载成功! (来自训练检查点)")
                if 'mAP' in checkpoint:
                    print(f"   训练时最佳mAP: {checkpoint['mAP']:.4f}")
                if 'epoch' in checkpoint:
                    print(f"   训练轮次: {checkpoint['epoch']}")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ 权重加载成功! (纯模型权重)")
                
        except Exception as e:
            print(f"⚠️ 权重加载失败，使用随机初始化: {e}")
            print("   这将导致检测结果不准确!")
    else:
        print(f"⚠️ 未找到权重文件: {WEIGHT_PATH}")
        print("   使用随机初始化权重 (检测结果无意义)")
    
    model.eval()
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 模型参数:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    
    # 3. 执行测试
    if TEST_MODE == "single":
        # 单张图片测试
        result = test_single_image(
            model=model,
            image_path=SINGLE_IMAGE_PATH,
            save_path=SINGLE_SAVE_PATH,
            device=device,
            conf_thres=CONF_THRES,
            iou_thres=IOU_THRES,
            class_names=CLASS_NAMES,
            img_size=IMG_SIZE
        )
        
        if result and result['num_detections'] > 0:
            print(f"\n🎉 测试完成! 检测到 {result['num_detections']} 个目标")
        
    elif TEST_MODE == "batch":
        # 批量测试
        results = test_batch_images(
            model=model,
            image_dir=BATCH_IMAGE_DIR,
            output_dir=BATCH_OUTPUT_DIR,
            device=device,
            conf_thres=CONF_THRES,
            iou_thres=IOU_THRES,
            class_names=CLASS_NAMES,
            img_size=IMG_SIZE,
            max_images=MAX_BATCH_IMAGES,
            image_list_file=test_txt
        )
    
    print("\n" + "="*60)
    print("✅ 测试完成!")
    print("="*60 + "\n")
