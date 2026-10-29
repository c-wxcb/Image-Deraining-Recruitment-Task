## 代码完善  
DerainNet：    
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
MyModel：  
```python
import torch.nn as nn

class mymoodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,3,1,1)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        block_list=[]
        for i in range(6):
            block_list.append(
                nn.Sequential(
                    nn.Conv2d(64,64,3,1,1),
                    nn.BatchNorm2d(64),
                )

            )# relu加在了后面
        self.blocks=nn.Sequential(*block_list)
        self.shortcut=nn.Identity()
        self.conv8=nn.Conv2d(64,3,3,1,1)

    def forward(self, x):
        initial_rained=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        for block in self.blocks:
            identity=self.shortcut(x)
            x=block(identity)
            x=self.relu(x+identity)
        rain_layer=self.conv8(x)
        rain_removed=initial_rained-rain_layer
        return rain_removed

```


get_args:
```python
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet", "DerainNet", "MyModel"], help="Model to use")


```
导入：
```python
from models.DerainNet import derainnet
from models.MyModel import mymoodel

```
判断：
```python
    elif args.model == "DerainNet":
        model = derainnet().to(device)
    elif args.model == "MyModel":
        model = mymoodel.to(device)

```