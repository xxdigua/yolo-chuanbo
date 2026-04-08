# YOLOv8 从底层架构实现

基于 PyTorch 从零实现的 YOLOv8 目标检测模型，包含完整的网络结构、数据加载、损失函数、训练和推理功能。

## 功能特性

- 完整的 YOLOv8 网络架构（Backbone + Neck + Head）
- 支持多种模型规格（n/s/m/l/x）
- TAL（Task Aligned Learning）标签分配策略
- DFL（Distribution Focal Loss）边界框回归
- 完整的训练、验证、测试流程
- mAP 评估指标计算
- 训练日志记录（CSV 格式）

## 项目结构

```
yolov8/
├── models/                 # 模型定义
│   ├── layers.py           # 基础层（Conv、DFL等）
│   ├── backbone.py         # 主干网络（CSPDarknet）
│   ├── neck.py             # 颈部网络（FPN+PAN）
│   ├── head.py             # 检测头（解耦头设计）
│   └── yolov8.py           # 完整模型定义
├── utils/                  # 工具函数
│   ├── dataloader.py       # 数据加载器
│   ├── loss.py             # 损失函数（VFL + DFL + CIoU）
│   ├── utils.py            # 后处理、mAP计算等
│   ├── tal.py              # TAL 标签分配
│   └── ops.py              # 基础运算
├── data/                   # 数据目录
│   └── VOCdevkit/          # 数据集
│       ├── images/         # 图像文件
│       ├── labels/         # 标注文件（YOLO格式）
│       ├── train.txt       # 训练集列表
│       ├── val.txt         # 验证集列表
│       └── test.txt        # 测试集列表
├── weights/                # 权重保存目录
├── logs/                   # 训练日志
├── debug_output/           # 训练可视化
├── val_output/             # 验证可视化
├── train1.py               # 训练脚本
├── test.py                 # 测试脚本
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- OpenCV
- NumPy
- tqdm

```bash
pip install torch torchvision opencv-python numpy tqdm
```

## 数据集格式

采用 YOLO 格式标注，每张图像对应一个 `.txt` 标注文件：

```
# 每行格式：class x_center y_center width height（归一化坐标）
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

数据集目录结构：
```
VOCdevkit/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── labels/
│   ├── 000001.txt
│   ├── 000002.txt
│   └── ...
├── train.txt    # 训练集图像列表
├── val.txt      # 验证集图像列表
└── test.txt     # 测试集图像列表
```

## 模型配置

支持以下模型规格：

| 模型 | depth_multiple | width_multiple | 参数量 |
|------|----------------|----------------|--------|
| YOLOv8n | 0.33 | 0.25 | ~3.2M |
| YOLOv8s | 0.33 | 0.50 | ~11.2M |
| YOLOv8m | 0.67 | 0.75 | ~25.9M |
| YOLOv8l | 1.00 | 1.00 | ~43.7M |
| YOLOv8x | 1.33 | 1.25 | ~68.2M |

## 训练

修改 `train1.py` 中的配置：

```python
num_classes = 6           # 类别数量
img_size = 640            # 输入尺寸
batch_size = 16           # 批次大小
epochs = 200              # 训练轮数

# 数据路径
data_root = r"D:\yolo\yolov8\data\VOCdevkit"
train_txt = os.path.join(data_root, "train.txt")
val_txt = os.path.join(data_root, "val.txt")
```

运行训练：

```bash
python train1.py
```

训练输出：
- `weights/last.pth` - 最新权重
- `weights/best.pth` - 最佳权重
- `logs/training_log.csv` - 训练日志
- `val_output/` - 验证可视化结果

## 测试

修改 `test.py` 中的配置：

```python
WEIGHT_PATH = "D:/yolo/yolov8/weights/last.pth"
NUM_CLASSES = 6

CLASS_NAMES = [
    'ore carrier', 'bulk cargo carrier', 'fishing boat',
    'general cargo ship', 'container ship', 'passenger ship'
]

CONF_THRES = 0.50  # 置信度阈值
IOU_THRES = 0.45   # NMS IoU阈值
```

运行测试：

```bash
python test.py
```

测试模式：
- `TEST_MODE = "single"` - 单张图片测试
- `TEST_MODE = "batch"` - 批量测试

## 推理示例

```python
import torch
import cv2
from models.yolov8 import yolov8_n
from utils.utils import postprocess

# 加载模型
model = yolov8_n(num_classes=6)
model.load_state_dict(torch.load('weights/best.pth')['model_state_dict'])
model.eval()

# 预处理
img = cv2.imread("test.jpg")
resized = cv2.resize(img, (640, 640))
rgb_img = resized[:, :, ::-1].astype(np.float32) / 255.0
input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0)

# 推理
with torch.no_grad():
    outputs = model(input_tensor)

# 后处理
bboxes = postprocess(outputs, conf_thres=0.5, iou_thres=0.45, img_size=640)

# 绘制结果
for box in bboxes[0]:
    x1, y1, x2, y2, conf, cls_id = map(int, box[:5].tolist()), box[5]
    cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("result.jpg", resized)
```

## 模型架构

### Backbone - CSPDarknet
- 使用 CSP 结构增强特征提取能力
- 支持 P3、P4、P5 三个尺度输出

### Neck - FPN+PAN
- FPN：自顶向下融合语义信息
- PAN：自底向上融合定位信息

### Head - Decoupled Head
- 分类和回归分支解耦
- DFL（Distribution Focal Loss）预测边界框分布
- 无锚框设计，直接预测目标位置

## 损失函数

- **分类损失**：Varifocal Loss (VFL)
- **回归损失**：CIoU Loss + DFL
- **标签分配**：Task Aligned Learning (TAL)

## 注意事项

1. 本项目主要用于学习 YOLOv8 的底层实现原理
2. 训练时图像会被直接 resize 到 640×640（不保持长宽比）
3. 测试时输出图像为 640×640 尺寸
4. 建议使用 GPU 进行训练

## 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [YOLOv8 论文](https://arxiv.org/abs/2305.09972)
- [TAL: Task Aligned Learning](https://arxiv.org/abs/2108.07755)
