from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    model = RTDETR(r"runs/detect/train5/weights/best.pt")
    model.val(
              split='val',
              imgsz=640,
              batch=4,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              save_txt=True,
              project='runs/val',
              name='exp',
              )