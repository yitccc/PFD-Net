from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = RTDETR(r"improve_multimodal/rtdetr/p2.yaml")  # 多模态模型
    # model.load('D:/BaiduNetdiskDownload/YOLO-multimodal/YOLO-multimodal/runs\detect/train/weights/best.pt')
    model.train(data=r"data.yaml",  # 数据集路径
                batch=4,
                epochs=300,
                amp=False,
                workers=8,
                optimizer='AdamW',
                lr0=0.001,
                device='0'
                )