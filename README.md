# AdaIN-pytorch-implement

![cheems](https://github.com/HuangXingjie2002/AdaIN-pytorch-implement/blob/main/image/READEME/d__github_AdaIN_AtentionAETransfer.png)

你需要安装如下python库


```python
# pytorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# or
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# opencv
pip install opencv-python
# or
conda install opencv-python

# 其他python基础库
```
*如果想自己实现一次的话，可以看AdaINImplement.md, 提供了一种实现思路*

#### 食用方法

1. 配置文件解读

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 判断是否有GPU
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
```
> 所有的可配置信息都在配置文件中显示，不论是模型的构建还是图片的风格迁移都在这里
> 1. 指定风格图片的地址`styleSrc`，`epochs`，`lr`， `trainingImageSize`用于构建模型
> 2. 指定内容图片的地址`contentSrc`, 风格迁移的参数`transformParamDir`, 用于生成图片

2. 模型的构建

在配置好`config.py`之后，直接运行`buildTransFormParam.py`文件，会自动构建该风格图片的风格解码器。
```python
>>> python buildTransFormParam.py
Loading Packages ...
Done
Loading style image ...
Done
Loading content image ...
Done
Loading VGG19 ...
Done
Decoder generate ...
Done
Build AdaINTransfer model ...
Done
Optimizing ...
[0]:    [Loss]: 2.472977876663208       [Content Loss]: 2.286891460418701       [Style Loss]: 0.6202879548072815
[20]:   [Loss]: 2.2495431900024414      [Content Loss]: 2.106981039047241       [Style Loss]: 0.4752069115638733
...

Done
Model has been saved, name: StarryNight-Transfer-Model1000_decoder.pt # 模型保存的名称
```

3. 风格图片的生成

在得到模型后，再在congfig.py中定义模型的地址`transformParamDir`, 随后指定输出图片的大小和内容图片的地址，
注意此时风格图片地址不可以改变，如果改变的话就是不同风格的融合了。  
运行`styleImageGenerate.py`  
```python
python styleImageGenerate.py       

Load Vgg ...
Done !
Initialize ...
Done !
Generating ...
Done !
```
图片会自动保存在你之前定义的位置`outputImageName`


