'''
Date: 2022-04-23 15:18:01
LastEditors: Mushr
LastEditTime: 2022-04-23 17:21:24
description: Do not edit
FilePath: \Poetry-Cloud\StyleTransForm\src\AtentionAETransfer\VideoFastTransform.py
'''
from config import *

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from torchvision.utils import save_image, Image
from torchvision.models import vgg19
import cv2
from matplotlib import pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = vgg19(pretrained=True).eval()

def vggExactor(image, vgg, layerName):
    module = nn.Sequential(

    )
    reluInd = 1
    others = 1
    for layer in vgg.features:
        if isinstance(layer, nn.Conv2d):
            name = f'conv_{others}'
            others += 1
        elif isinstance(layer, nn.ReLU):
            name = f'ReLU_{reluInd}'
            reluInd += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = f'MaxPool2d_{others}'
            others += 1
        else:
            raise f"Error {layer.__name__}"
        
        module.add_module(name, layer)
        if name == layerName:
            outPut = module(image)
            return outPut


def factoryOfDecoder(device):
    decoder = nn.Sequential(
        # 输出尺寸为 [1, 256, 128, 128]

        nn.ReflectionPad2d((1, 1, 1, 1)),  # 增加一层
        nn.Conv2d(256, 256, (3, 3)),  # 下降一层
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        # [1, 128, 128, 128]
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReLU(),
        # [1, 128, 256, 256]
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.Upsample(scale_factor=2, mode='nearest'),
        # [1, 128, 512, 512]
        nn.ReLU(),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 32, (3, 3)),
        nn.ReLU(),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 16, (3, 3)),
        nn.ReLU(),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 3, (3, 3)),
        nn.ReLU(),
        # [1, 3, 512, 512]
    )
    return decoder.to(device)


def loadImage(src, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image = transform(Image.open(src)).unsqueeze(0)
    image.requires_grad = False
    return image


def getAda(imageFeatureMap):
    meanVal = imageFeatureMap.mean(dim=(2, 3), keepdim=True)
    stdVal = imageFeatureMap.std(dim=(2, 3), keepdim=True)
    return meanVal, stdVal


def ICS(content, style, vgg):
    contentFeatures = vggExactor(content, vgg, 'ReLU_5')
    styleFeatures = vggExactor(style, vgg, 'ReLU_5')
    meanC, stdC = getAda(contentFeatures)
    meanS, stdS = getAda(styleFeatures)

    return  (contentFeatures - meanC).div(stdC) * stdS + meanS



class Ada:
    def __init__(self, decoder, style, vgg):
        self.decoder = decoder
        self.style = style
        self.vgg = vgg
    
    def __ICS(self, content):
        return ICS(content, self.style, self.vgg)
    
    def __call__(self, content):
        return self.decoder(self.__ICS(content))


def imshow(image):
    if image.requires_grad:
        image = image.detach()
    image = image.cpu()
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    image = image.numpy()
    plt.imshow(image)
    plt.show()


class AdaTransFormer:
    def __init__(self, styleImageSrc, decoderParamSrc, vgg, device, imageSize):
        self.style = loadImage(styleImageSrc, imageSize).to(device)
        self.decoder = factoryOfDecoder(device)
        self.decoder.load_state_dict(torch.load(decoderParamSrc))
        self.decoder = self.decoder.eval().to(device)
        self.vgg = vgg.eval().to(device)
        self.device = device
        self.adam = Ada(self.decoder, self.style, self.vgg)
    
    def __call__(self, inputImageSrc):
        if isinstance(inputImageSrc, str):
            content = loadImage(inputImageSrc, self.style.shape[-2]).to(self.device)
        else:
            content = inputImageSrc.to(self.device)
        return self.adam(content)


starNightTransFer = AdaTransFormer(styleSrc, transformParamDir, vgg, device, 512)


capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame,1)   #镜像操作
    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1).unsqueeze(0)
    frame = frame.to(torch.float)
    frame = frame.to(device)
    frame = starNightTransFer(frame)
    frame = frame.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    cv2.imshow("video", frame)
    key = cv2.waitKey(50)
    #print(key)
    if key  == ord('q'):  #判断是哪一个键按下
        break
cv2.destroyAllWindows()
