import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  model = YOLO(r'G:\deeplearning\github\ultralytics-original\P2small-CBAM-BiFPN-D.yaml')
  model.load(r'G:\deeplearning\github\ultralytics-original\yolo11n.pt')  #注释则不加载
  results = model.train(
    data=r'G:\deeplearning\ultralytics-8.3.163\NEU-data11\NEU-DET\NEU-DET.yaml',  #数据集配置文件的路径
    epochs=300,  #训练轮次总数
    batch=12,  #批量大小，即单次输入多少图片训练
    imgsz=640,  #训练图像尺寸
    workers=0,  #加载数据的工作线程数
    device= 0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='auto',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
    amp= True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
    cache=False,  # True 在内存中缓存数据集图像，服务器推荐开启
    resume=False,  # True 继续训练，False 重新训练
    patience=50,  # 早停，即训练过程中，如果验证集指标不再提升，则停止训练
    cos_lr=False,  # 开启余弦退火（比线性衰减收敛更稳）
)