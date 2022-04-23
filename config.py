'''
Date: 2022-04-19 20:28:36
LastEditors: Mushr
LastEditTime: 2022-04-23 17:46:59
description: Do not edit
FilePath: \AdaIN\config.py
'''
import torch

device = 'cpu' # 判断是否有GPU
styleName = 'Monet-Transfer-Model' 
transformParamDir = './preTrainModels/monet.pt' # 参数文件
styleLayers = ['ReLU_1', 'ReLU_2', 'ReLU_3'] # 特征提取的维度, 可以自由选择
styleWeight = 0.3
trainingImageSize = 256 # 推荐设置小一点，因为计算资源不足
outputImageSize = 1024 # 输出图片维度, 视情况而定
outputImageName = './data/test/output.jpg'
lr = 0.0001 # 如果LOSS 不下降的话，需要把学习率降低
epochs = 1000 # 训练次数

styleSrc = './data/ori/monet.jpg'             # 风格图片地址
contentSrc = './data/ori/lightHouse.jpg'      # 内容图片地址
