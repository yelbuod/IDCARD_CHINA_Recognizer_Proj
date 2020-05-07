# -*- coding: utf-8 -*-
"""
@author: Minus
"""
from imutils.video import FPS
from PIL import Image
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--image-file", required=True,
    help="the path of input image file")
args = vars(ap.parse_args())

fps = FPS().start()

from config import *
from crnn.network_torch import CRNN
from application import idcard
print('[INFO] Successfully import from factory...')

# 导入基于yolo的keras文本检测
# 设置keras-yolo属性和进程
if GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
    import tensorflow as tf
    from keras import backend as K
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.3## GPU最大占用量
    config.gpu_options.allow_growth = True##GPU是否可动态增加
    K.set_session(tf.Session(config=config))
    K.get_session().run(tf.global_variables_initializer())
else:
    ##CPU启动
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
print('[INFO] Successfully set keras-yolo config and session...')

scale,maxScale = IMGSIZE[0],2048
# 导入文字检测模块
from text.keras_detect import text_detect
# 导入方向检测模块
from text.opencv_dnn_detect import angle_detect
print('[INFO] Successfully import text detect and angle detect...')
# 基于Torch-CRNN的ocr
if ocr_redis:
    ##多任务并发识别
    from apphelper.redisbase import redisDataBase
    ocr = redisDataBase().put_values
else:
    from crnn.keys import alphabetChinese,alphabetEnglish
    if chineseModel:
        alphabet = alphabetChinese
        if LSTMFLAG:
            ocrModel = ocrModelTorchLstm
        else:
            ocrModel = ocrModelTorchDense
            
    else:
        ocrModel = ocrModelTorchEng
        alphabet = alphabetEnglish
        LSTMFLAG = True
    
    nclass = len(alphabet)+1
    crnn = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
    print('[INFO] Successfully initialize CRNN recognizer')

    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
        print("[INFO] Successfully load Torch-ocr model...")
    else:
        print("download model or tranform model with tools!")
        
    ocr = crnn.predict_job
    
if __name__ == "__main__":

    from main import TextOcrModel
    # initialize model pipeline
    model =  TextOcrModel(ocr,text_detect,angle_detect)
    # arguments
    textAngle = False
    # load image
    img = cv2.imread(args["image_file"])

    # predict
    detectAngle = textAngle
    result,angle= model.model(img,
                                scale=scale,
                                maxScale=maxScale,
                                detectAngle=detectAngle,##是否进行文字方向检测，通过web传参控制
                                MAX_HORIZONTAL_GAP=100,##字符之间的最大间隔，用于文本行的合并
                                MIN_V_OVERLAPS=0.6,
                                MIN_SIZE_SIM=0.6,
                                TEXT_PROPOSALS_MIN_SCORE=0.1,
                                TEXT_PROPOSALS_NMS_THRESH=0.3,
                                TEXT_LINE_NMS_THRESH = 0.99,##文本行之间测iou值
                                LINE_MIN_SCORE=0.1,
                                leftAdjustAlph=0.01,##对检测的文本行进行向左延伸
                                rightAdjustAlph=0.01,##对检测的文本行进行向右延伸
                                )

    res = idcard.idcard(result)
    # res.res 是idcard类的一个用于存放各类信息集合的列表属性
    res = res.res
    res =[ {'text':res[key],'attribute':key,'box':{}} for key in res]
    print(res)
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))