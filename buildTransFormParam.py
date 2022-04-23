'''
Date: 2022-04-23 15:24:41
LastEditors: Mushr
LastEditTime: 2022-04-23 16:26:06
description: Do not edit
FilePath: \Poetry-Cloud\StyleTransForm\src\AtentionAETransfer\buildTransFormParam.py
'''
'''
Date: 2022-04-23 15:24:41
LastEditors: Mushr
LastEditTime: 2022-04-23 15:33:49
description: Do not edit
FilePath: \Poetry-Cloud\StyleTransForm\src\AtentionAETransfer\buildTransFormParam.ipy
'''

# 配置文件
from config import *

# 导包

print('Loading Packages ...')
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg19
from torchvision.utils import save_image, Image
from torchvision import transforms  # 转换方法|

from matplotlib import pyplot as plt

print('Done')

# 图片导入
def imageLoader(imageSrc: str, imageSize: int, device) -> torch.tensor:
    transform = transforms.Compose([
        transforms.Resize(imageSize),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor()
    ])

    image = Image.open(imageSrc)
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)



# 从vgg中提取特征
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


# 创建AdaIN单元
def adaINNormal(X: torch.tensor, dim: Union[int, tuple], keepdim=True) -> Tuple[torch.tensor]:

    meanVal = X.mean(dim=dim, keepdim=True)
    stdVal = X.std(dim=dim, keepdim=True)
    return meanVal, stdVal

class AdaIN:
    def __init__(self, content, style):
        super(AdaIN).__init__()
        self.content = content
        self.style = style
        contentMeanShift, contentStdShift = adaINNormal(
            self.content, dim=(2, 3), keepdim=True)
        styleMeanShift, styleStdShift = adaINNormal(
            self.style, dim=(2, 3), keepdim=True)
        # 计算内容与风格图片的平均值和标准差

        # 计算内容图片的风格迁移
        
        self.FCS = styleStdShift * \
            ((self.content - contentMeanShift).div(contentStdShift)) + styleMeanShift
        self.FCS = self.FCS.to(device)
    def __call__(self):
        return self.FCS # 返回ICS单元

# 内容损失与风格损失
class ContentLoss(nn.Module):
    def __init__(self, ICS):
        super(ContentLoss, self).__init__()
        self.ICS = ICS
        self.loss = nn.MSELoss()
        self.ICloss = 0
    def forward(self, inputFeaturemap):
        self.ICloss = self.loss(self.ICS, inputFeaturemap)
        # self.ICloss = self.ICloss.detach()
        # self.RecentInput = inputFeaturemap
        # print(inputFeaturemap)
        # print(self.ICloss)
        return inputFeaturemap # 不更新参数，只是起记录作用

class StyleLoss(nn.Module):
    def __init__(self, Style):
        # 风格损失记录下同一层的风格损失即可
        super(StyleLoss, self).__init__()
        self.Style = Style
        self.styleMean, self.styleStd = adaINNormal(Style, dim=(2, 3), keepdim=True)
        self.loss = nn.MSELoss()
        self.ISloss = 0
    
    def forward(self, inputImage):
        inputMean, inputStd = adaINNormal(inputImage, dim=(2, 3), keepdim=True)
        # print(inputMean.shape, inputStd.shape, self.styleMean.shape, self.styleStd.shape)
        self.ISloss = self.loss(inputMean, self.styleMean) + self.loss(inputStd, self.styleStd)
        # self.ISloss = self.ISloss.detach()
        return inputImage # 不做修改直接向后传递


print('Loading style image ...')
style = imageLoader(styleSrc, trainingImageSize, device)
print('Done')

print('Loading content image ...')
content = imageLoader(styleSrc, trainingImageSize, device)
print('Done')

print('Loading VGG19 ...')
vgg = vgg19(pretrained=True).eval().to(device)
print('Done')

print('Decoder generate ...')
decoder = nn.Sequential(
    # 输出尺寸为 [1, 256, 128, 128]

    nn.ReflectionPad2d((1, 1, 1, 1)), # 增加一层
    nn.Conv2d(256, 256, (3, 3)), # 下降一层
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

print('Done')
decoder.to(device) # 移动到某个设备中


# 创建AdaIN风格迁移单元
class AdaINTransfer(nn.Module):
    def __init__(self, content, style, decoder, cnn, styleLossLayer: List[str], alpha, device):
        super(AdaINTransfer, self).__init__()
        self.alpha = alpha
        self.decoder = decoder

        # 计算得到特征
        # ReLU_5
        adaFeatureLayer = 'ReLU_5'
        self.ics = AdaIN(vggExactor(content, cnn, adaFeatureLayer), vggExactor(style, cnn, adaFeatureLayer))()
        self.ics.to(device)
        self.ics.require_grad = False # 不允许更新
        self.styleLosses = []
        self.contentLosses = []
        # 构建loss判别器

        self.module = nn.Sequential()
        reluInd = 1
        otherInd = 1
        for layer in cnn.features:
            if isinstance(layer, nn.ReLU):
                name = f'ReLU_{reluInd}'
                reluInd += 1
                layer = nn.ReLU(inplace=False)
            else:
                name = f'Other_{otherInd}'
                otherInd += 1
            
            self.module.add_module(name, layer)

            if name in styleLossLayer:
                # 如果当前层在风格损失计算范围内，那么
                currentStyle = self.module(style) # 记录当前的风格
                sl = StyleLoss(currentStyle) # 构建一个损失函数，保存当前层的风格
                self.module.add_module(f'StyleLoss_{otherInd}', sl) # 添加这个风格层，当下次调用module的时候就会更新这个参数
                self.styleLosses.append(sl)
                otherInd += 1
            if name == adaFeatureLayer:
                # print(name)
                cl = ContentLoss(self.ics) # 使用ICS得到内容损失函数，使得最终生成的图片的特征尽可能靠近 ics
                self.module.add_module(f'ContentLoss_{otherInd}', cl)
                self.contentLosses.append(cl)
                otherInd += 1
        
        for param in self.module.parameters():
            param.requires_grad = False # MD 
    def forward(self, image):
        # 模型在初始化的时候就已经完成了ics的构建，此时输入的图片是ics，并且将一直是ics
        
        # 解码器得到图片
        image = self.decoder(image) 
        # print(image.shape)
        # 把解码器得到的图片输入到判别器中

        self.module(image) # 不需要管输出是什么

        cl = 0
        sl = 0

        for layer in self.contentLosses:
            cl += layer.ICloss
        
        for layer in self.styleLosses:
            sl += layer.ISloss
        
        total = cl + self.alpha * sl

        return total, cl, sl


print('Build AdaINTransfer model ...')
net = AdaINTransfer(content, style, decoder, vgg, styleLayers, styleWeight, device)
print('Done')



print('Optimizing ...')
imageSample = net.ics.detach() # 复制一份
optimizer = optim.Adam(decoder.parameters(), lr=lr)
decoder.to(device)
styleLossRecord = []
contentLossRecord = []
totalLossRecord = []
for epoch in range(epochs):
    image = decoder(imageSample)
    # 把imageSample编码为 content
    net.module(image)
    

    cl = 0
    sl = 0
    
    for layer in net.contentLosses:
        cl += layer.ICloss
    for layer in net.styleLosses:
        sl += layer.ISloss
    # print(cl, sl, net.alpha)
    total = cl + net.alpha * sl
    optimizer.zero_grad()
    total.backward(retain_graph=True)
    optimizer.step()

    if epoch % 20 == 0:
        print(f'[{epoch}]:\t[Loss]: {total.item()}\t[Content Loss]: {cl.item()}\t[Style Loss]: {sl.item()}')
        styleLossRecord.append(sl.item())
        contentLossRecord.append(cl.item())
        totalLossRecord.append(total.item())
    

plt.figure(figsize=(10, 5))
plt.plot(styleLossRecord, label='Style Loss')
plt.plot(contentLossRecord, label='Content Loss')
plt.plot(totalLossRecord, label='Total Loss')
plt.legend()
plt.show()
plt.savefig(f'{styleName}{epochs}.png')

print('Done')


torch.save(decoder.state_dict(), f'{styleName}{epochs}_decoder.pt')
print(f'Model has been saved, name: {styleName}{epochs}_decoder.pt')
