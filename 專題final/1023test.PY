import numpy as np
import cv2
import time
from grabscreen import grab_screen
from PIL import Image
from yolo import YOLO
import tensorflow as tf
 
import colorsys
import os

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8   #爆显存的话可以在此调整程序的显存占用情况
session = tf.Session(config=config)
 
yolo = YOLO()


# 创建创建一个存储检测结果的dir
result_path = './result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# result如果之前存放的有文件，全部清除
for i in os.listdir(result_path):
    path_file = os.path.join(result_path,i)  
    if os.path.isfile(path_file):
        os.remove(path_file)

#创建一个记录检测结果的文件
txt_path =result_path + '/result.txt'
file = open(txt_path,'w')  


 
while True:
 
    image_array = grab_screen(region=(0, 0, 1280, 720))
    # 获取屏幕，(0, 0, 1280, 720)表示从屏幕坐标（0,0）即左上角，截取往右1280和往下720的画面
    array_to_image = Image.fromarray(image_array, mode='RGB') #将array转成图像，才能送入yolo进行预测
    img = yolo.detect_image(array_to_image)  #调用yolo文件里的函数进行检测
 
    img = np.asarray(img) #将图像转成array
 
    cv2.imshow('window',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))#将截取的画面从另一窗口显示出来，对速度会有一点点影响，不过也就截取每帧多了大约0.01s的时间
    
    file.write('  score: '+str(yolo.detect_image.score)+' \nlocation: top: '+str(top)+'、 bottom: '+str(bottom)+'、 left: '+str(left)+'、 right: '+str(right)+'\n')
    
    
    
    if cv2.waitKey(25) & 0xFF == ord('q'):  #按q退出，记得输入切成英语再按q
        cv2.destroyAllWindows()
        break
