import torch.nn as nn

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
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

        return None
