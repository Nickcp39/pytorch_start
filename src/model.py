

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
        # 32* 32 的input
        self.model= nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(3, 32, 5, 1, 2),
            # kernel size 2*2 最大池化
            nn.MaxPool2d(2),

            # input 32, output 32, kernel size 5 , thread 1, padding 2 =  不知道为什么padding 是2
            nn.Conv2d(32, 32, 5, 1, 2),
            #再来个最大池化
            nn.MaxPool2d(2),
            # input 32, output 65 , kernel 5, thread 1 , padding 2
            nn.Conv2d(32, 64, 5, 1, 2),
            # 最大池化
            nn.MaxPool2d(2),
            #展平
            nn.Flatten(),
            # 占平城 fully connected 64 * 4 * 4 的size
            nn.Linear(64 * 4 * 4, 64),
            # input 64 , output 10
            nn.Linear(64, 10, )
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    # 可以在这里测试网络模型的正确性
    #先创造一个网络模型
    tudui = Tudui()
    # 然后确定输入尺寸
    input = torch.ones((64,3,32,32))
    # 检查输出尺寸就可以了
    output = tudui(input)
    print(output.shape)

