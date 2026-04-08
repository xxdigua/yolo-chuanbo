import torch
from models.yolov8 import yolov8_n
from utils.dataloader import get_dataloader
from utils.utils import postprocess, plot_bboxes
import cv2

# 加载模型
model = yolov8_n()
model.eval()

# 加载训练好的权重
try:
    model.load_state_dict(torch.load('yolov8_coco.pth'))
    print("Loaded trained weights")
except:
    print("No trained weights found, using random initialization")

# 获取数据加载器
test_loader = get_dataloader(
    img_dir="D:/yolo/yolov8/data/coco128/images/train2017",
    label_dir="D:/yolo/yolov8/data/coco128/labels/train2017",
    batch_size=1,
    img_size=640,
    augment=False,
    shuffle=False
)

# 测试预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_loader):
        if i >= 1:  # 只测试一张图片
            break
        
        imgs, labels = imgs.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(imgs)
        
        # 后处理获取预测结果
        batch_preds = postprocess(outputs, conf_thres=0.01, iou_thres=0.45, reg_max=16)
        
        # 查看预测结果
        print(f"Number of predictions: {len(batch_preds[0])}")
        if len(batch_preds[0]) > 0:
            print("Predictions:")
            for pred in batch_preds[0][:5]:  # 显示前5个预测
                print(f"  Box: {pred[:4]}, Conf: {pred[4]:.4f}, Class: {int(pred[5])}")
        
        # 可视化预测结果
        img = imgs[0].cpu().permute(1, 2, 0).numpy() * 255
        img = img.astype('uint8')
        
        if len(batch_preds[0]) > 0:
            img = plot_bboxes(img, batch_preds[0])
        
        # 保存结果
        cv2.imwrite('test_prediction.jpg', img)
        print("Prediction saved to test_prediction.jpg")