### 代码  
```python
import torch.nn as nn
import torch
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        out=self.conv(x)
        out=out+x
        return out

class derainnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.MaxPool2d(2,2)
        self.branch_1 = nn.Sequential(
            *[Block(in_channels=64, out_channels=64) for _ in range(8)],
        )
        self.branch_2 = nn.Sequential(
            *[Block(in_channels=64, out_channels=64) for _ in range(8)],
        )
        self.branch_3 = nn.Sequential(
            *[Block(in_channels=64, out_channels=64) for _ in range(8)],
        )
        self.conv2=nn.Conv2d(192,3,3,1,1)


    def forward(self,x):
        x0=self.conv_1(x)
        x1=x0
        x2=self.downsample(x0)
        x3=self.downsample(x2)
        out1=self.branch_1(x1)
        out2=self.branch_2(x2)
        out3=self.branch_3(x3)
        out1=out1+x1
        out2=out2+x2
        out3=out3+x3
        target_size=out1.size()[-2:]
        out2=F.interpolate(out2,size=target_size,)
        out3=F.interpolate(out3,size=target_size,)
        out=torch.cat([out1,out2,out3],1)
        rain_layer=self.conv2(out)
        rain_removed=x-rain_layer

        return rain_removed
```  
### 运行 
遗憾的就是我的5060不支持我运行，说我显存不足，关了wallpaper也不够：  
```python
torch.AcceleratorError: CUDA error: out of memory

```

### 思考  
   1.关于下采样  
   首先就是去雨有很多要求，既要抓住雨丝的特征，也要尽量多的感受野，这就有了矛盾，抓细节，你就没法看全貌，看全貌，细节就容易丢，所以干脆直接分开干活，分出来三个分支，一个在原分辨率下抓细节，另外两个就去抓全貌，那么怎么实现，具体就是使用下采样，某一块里面取最大值（最大池化），这能缩小图片尺寸，能阔大感受野，能够实现“看全貌”的特性  
   关于下采样的技术细节，有最大池化，也就是某一块里取最大值，还有平均池化，某一块里取均值
  2.关于上采样  
  最后我们需要的是原分辨率的雨层，下采样之后的图片分辨率降低，一张256x256的图片可能变成64x64的图片，要输出雨层自然要恢复原先的分辨率，所以要完成上采样  
  关于技术细节，我采用的是`F.interpolate()`方法，具体怎么复原的，常见的是双线性插值，根据举例加权填充，还有直接复制的，直接填充相邻的值  
  3.关于通道维度上拼接
  我们希望把最后三个分支的结果综合起来，但是到底该怎么综合我这里产生了疑惑，为什么不能逐像素相加，后面发现这样的话会把每个分支的信息混合而不是综合，无法区分贡献度，但是按照通道维度拼接就体现出来每个分支的贡献度（这也是相加做不到的）