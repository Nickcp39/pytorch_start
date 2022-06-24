

import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model= nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10, )
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    tudui = Tudui()
    input = torch.ones((64,3,32,32))
    output =  tudui(input)
    print(output.shape )
    """
    output  = torch.Size([64, 10])
    64个数据， 每个数据上边10个数据点， 10个数据代表每个图片在10个不同的类别中， 的存在概率。 picture01  = [0.1,0.1,0.1,。。。。。。1] =  意味着p1 图对于分类在10个中的概率分别都为0.1 
    """