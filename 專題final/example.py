import numpy as np
import cv2
import time
from grabscreen import grab_screen
from PIL import Image
from yolo import YOLO
import tensorflow as tf
 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8   #爆显存的话可以在此调整程序的显存占用情况
session = tf.Session(config=config)
 
yolo = YOLO()
 
while True:
 
    image_array = grab_screen(region=(205, 130, 1280, 1260))
    # 获取屏幕，(0, 0, 1280, 720)表示从屏幕坐标（0,0）即左上角，截取往右1280和往下720的画面
    array_to_image = Image.fromarray(image_array, mode='RGB') #将array转成图像，才能送入yolo进行预测
    img = yolo.detect_image(array_to_image)  #调用yolo文件里的函数进行检测
 
    img = np.asarray(img) #将图像转成array
 
    cv2.imshow('window',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  #将截取的画面从另一窗口显示出来，对速度会有一点点影响，不过也就截取每帧多了大约0.01s的时间
    if cv2.waitKey(25) & 0xFF == ord('q'):  #按q退出，记得输入切成英语再按q
        cv2.destroyAllWindows()
        break