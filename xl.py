from ultralytics import YOLOv10
import os
import torch

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == '__main__':
    #预测的代码
    # model = YOLOv10('yolov10n.pt') #这里写上预测模型的路径
    # model.predict("ultralytics/assets/bus.jpg",save=True) #这里写预测资源的路径，可以是图片、视频或者摄像头
    # print(torch.cuda.is_available()) #输出是否为GPU检测，不用动


    #训练的代码
    model = YOLOv10('yolov10n.pt')  #这里写权重模型的位置
    model.train(data=r"D:\YOLOv10\dataset\data.yaml", epochs=200,batch=1,imgsz=640) #这里写训练参数位置