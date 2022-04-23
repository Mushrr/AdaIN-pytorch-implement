# AdaIN 模型实现指南 v-2

![image-20220421203531587](D:\Code\2022\Poetry-Cloud\StyleTransForm\src\AtentionAETransfer\image\READEME\image-20220421203531587.png)

##  模型组成

> 1. 模型有两个预先训练好的VGG模型，可以使用pytorch直接拿到
>
> 2. AdaIN单元，用于进行AdaIN层风格与内容的融合
> 3. 解码器，用于把经过AdaIN单元融合的特征解码为一张新的图片

## 模型构建过程

> 1. 通过风格图片$I_s$, 内容图片$I_c$，的输入，从VGG的某一个层中提取出特征输出 $\phi(I_s)\phi(I_c)$
> 2. 将得到的特征在AdaIN单元进行融合
> 3. 融合后的特征进入解码器
> 4. 解码器得到的图片再次进入VGG中，提取特征，并且计算内容于风格的相似度。

## 模型的难点

### 1. AdaIN单元如何完成转换？

* 对于VGG提取好的特征 $\phi(I_c)\phi(I_s)$经过如下操作完成特征的融合
* $I_c:  [Batch, C, H, W]$

$$
\mu = \frac{1}{HW}\sum_{h=1}^H\sum_{w=1}^Wx_{bchw} \\
\sigma = \frac{1}{HW}\sum_{h=1}^H\sum_{w=1}^W(x_{bchw} - \mu)^2 \\
AdaIN(I_c, I_s) = \sigma_s\frac{I_c - \mu_c}{\sigma_c} + \mu_s
$$

* 简单来说就是把每个batch，每个通道下的宽高为W, H的图片求均值，求方差，并记录下来。
* $AdaIN(I_c, I_s): [Batch, C, 1, 1]$

```python
A = torch.rand((1, 128, 100, 100))
meanA = A.sum(dim=(2, 3), keepdim=True)
stdA = A.std(dim=(2, 3), keepdim=True) # 保留维度
```



### 2. 第一次提取特征何时结束？

* 简易提取的越深越好，经历使得特征的通道数增多，在多个通道上进行特征融合效果会更好。
* 只需要提取一次就可以结束了，第一层VGG网络只是负责把两数据提取出来，后续可以把这些特征存储起来作为下面数个epoch的输入以更新`decoder`



### 3. 解码器什么结构？

> 不妨看看原文中如何描述
>
> *`The decoder mostly mirrors the encoder, with all pooling layers replaced by nearest up-sampling to reduce checkerboard effects. We use reflection padding in both f and g to avoid border artifacts. Another important architectural choice is whether the decoder should use instance, batch, or no normalization layers.`*
>
> 解码器大多数时候是编码器的镜像，所有的$pooling$操作转为最近上采样以减少棋盘效应



### 4. 损失函数如何定义与计算？

* 风格损失函数

$$
L_s = \sum_{i=1}^L||\mu(\phi_i(g(t)) - \mu(\phi_i(s))||_2 + ||\sigma(\phi_i(g(t))) -\sigma(\phi_i(s))||
$$

* 内容损失函数

$$
L_c = ||f(g(t)) - t||_2
$$

$$
L_{total} = L_c + \lambda L_s
$$

其中
$$
t = AdaIN(f(c), f(s)) \\
g 为解码器 \\
T(c, s) = g(t): 解码器的结果认为与生成的风格图片相似
$$


### 5. 解码器如何配置？

* `ConvTranspose2d` 用于转置卷积，降低图片的通道数
* `UpsamplingNearest2d`用于上采样，提升图片的维度

* 使用`ReLU`连接两个相邻的转置卷积上采样

$$
[B, C, H_{input}, W_{input}] \rightarrow [B, 3, H_{output}, W_{output}] \\
$$



* 当遇到通道数为`3`或者宽高达到$H_{output}$的时候，可以选择终止。
* 所以最终解码器的配置可以按如下配置

```python
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
```

```python
decoderImage = decoder(adaINFeature) # 输入经过AdaIN单元处理后的特征
```

### 6. 损失函数如何构建？

> 整个模型中变的部分是整个模型的参数，不变的是*`AdaIN`*单元
>
> 每次经过decoder之后总会得到一张新的图片，而这张图片将作为输入被丢入到VGG中与风格，内容构建损失函数。可以直接借助Gayts的网络进行训练

* 其中特征提取后的内容特征尽量与`AdaIN`单元相近*层取相同的层*
* 风格层需要自定义，把风格图片与`ICS`求均值方差损失函数

* VGGLoss初始化的时候可以把AdaIN的结果，风格图片一起输入，记录作为初始化
* 训练的时候构建在需要的风格和内容层计算LOSS，并记录下来返回

*内容损失*

```python
class ContentLoss(nn.Module):
    def __init__(self, ICS):
        super(ContentLoss, self).__init__()
        self.ICS = ICS
        self.loss = nn.MSELoss()
        self.ICloss = 0
    def forward(self, inputFeaturemap):
        self.ICloss = self.loss(self.ICS, inputFeaturemap)
        # print(inputFeaturemap)
        # print(self.ICloss)
        return inputFeaturemap # 不更新参数，只是起记录作用
```

*样式损失*

```python
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
        return inputImage # 不做修改直接向后传递
```

*vgg提取损失*

```python
class VGGLoss(nn.Module):
    def __init__(self, ICS, IS, contentLayer: str, styleLayers: List[str], cnn, alpha=1):
        super(VGGLoss, self).__init__()
        self.ICS = ICS # 经过AdaIN处理之后的特征层
        self.IS = IS # 输入的风格图片
        self.alpha = alpha
    
        self.cnn = nn.Sequential()
        reluInd = 1
        otherInd = 1

        self.contentLossLayer = []
        self.styleLossLayer = []
        for i, layer in enumerate(cnn.features):
            if isinstance(layer, nn.ReLU):
                name = f'ReLU_{reluInd}'
                reluInd += 1
            else:
                name = f'Other_{otherInd}'
                otherInd += 1

            self.cnn.add_module(name, layer)
            if (name == contentLayer):
                cl = ContentLoss(ICS)
                self.cnn.add_module(f'ContentLoss_{otherInd}', cl)
                otherInd += 1
                self.contentLossLayer.append(cl)
            elif (name in styleLayers):
                currentStyleFeature = self.cnn(IS)
                print(currentStyleFeature.shape)
                sl = StyleLoss(currentStyleFeature)
                self.cnn.add_module(f'StyleLoss_{otherInd}', sl)
                otherInd += 1
                self.styleLossLayer.append(sl)
        pass

    def forward(self, inputImage):
        outPut = self.cnn(inputImage) # 输入
        contentLoss = 0
        styleLoss = 0

        for contentlayer in self.contentLossLayer:
            contentLoss += contentlayer.ICloss
        
        for stylelayer in self.styleLossLayer:
            styleLoss += stylelayer.ISloss

        totalLoss = contentLoss + self.alpha * styleLoss
        return totalLoss, contentLoss, styleLoss # 返回风格层，内容层，总损失
        
```

* 一次前向传播完，可以通过自己构建的cnn网络中的`contentLoss`， `styleLoss`层更新`loss`
* 最终返回所有的`loss`,梯度下降。

### 7.  如何训练？

1. 确定风格样式图片 $I_C$, $I_S$

2. 输入提取一个`AdaIN`单元， 并得到$I_{CS}$
3. 进行迭代训练`decoder`















