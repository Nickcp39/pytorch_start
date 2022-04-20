
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 网络当中第一个卷积层

        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    # 定义一个 卷积forward函数 来对图像进行处理，
    # return X
    def forward(self, x):
        x = self.conv1(x)
        return x

# 初始化这个网络
tudui = Tudui()
print(tudui)
# 来看网络结构
writer = SummaryWriter("../logs")

step = 0

for data in dataloader:
    # 从data中拿到图像数据 来训练
    imgs, targets = data
    # 返回一个ouput 从网络中
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]
    # 6 个channel 是不能显示的， 只能用3个channel。 所以需要reshape output
    # 修改 output 的尺寸才可以继续操作数据

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1

writer.close()

