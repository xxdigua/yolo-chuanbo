import json
import os

# COCO 类别映射
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def coco2yolo(coco_json_path, img_dir, output_label_dir):
    """将COCO格式的标注转换为YOLO格式"""
    # 创建输出目录
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 加载COCO JSON文件
    try:
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error loading COCO JSON file: {e}")
        return
    
    # 创建图像ID到信息的映射
    img_info = {}
    for img in coco_data.get('images', []):
        img_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }
    
    # 按图像ID分组标注
    annotations_by_img = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)
    
    # 处理每个图像的标注
    processed_count = 0
    for img_id, anns in annotations_by_img.items():
        if img_id not in img_info:
            continue
        
        img_file = img_info[img_id]['file_name']
        img_w = img_info[img_id]['width']
        img_h = img_info[img_id]['height']
        
        # 生成YOLO格式的标签文件
        label_file = os.path.join(output_label_dir, img_file.replace('.jpg', '.txt'))
        try:
            with open(label_file, 'w', encoding='utf-8') as f:
                for ann in anns:
                    # 获取类别ID
                    category_id = ann['category_id'] - 1  # COCO类别ID从1开始，YOLO从0开始
                    
                    # 获取边界框
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # 转换为YOLO格式：x_center, y_center, width, height (归一化到0-1)
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    
                    # 写入标签文件
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            processed_count += 1
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"Conversion completed! Processed {processed_count} images.")

    print(f"Output labels saved to {output_label_dir}")

if __name__ == "__main__" :
    # 转换训练集
    coco2yolo(
        coco_json_path="d:\\yolo\\yolov8\\data\\coco\\annotations\\instances_train2017.json",
        img_dir="d:\\yolo\\yolov8\\data\\coco\\train2017",
        output_label_dir="d:\\yolo\\yolov8\\data\\coco\\labels_train2017"
    )
    
    # 转换验证集
    coco2yolo(
        coco_json_path="d:\\yolo\\yolov8\\data\\coco\\annotations\\instances_val2017.json",
        img_dir="d:\\yolo\\yolov8\\data\\coco\\val2017",
        output_label_dir="d:\\yolo\\yolov8\\data\\coco\\labels_val2017"
    )
  