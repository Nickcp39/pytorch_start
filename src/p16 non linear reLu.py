

# 对输入图像进行填充的基础方式
# 卷积层 conv2d 也可以做
# 非线性激活 Relu 层
# 引入一些非线性的特质 给神经网络network
#
import torch
import torchvision.datasets
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                     [-1, 3 ]])
input =  torch.reshape(input, (-1,1,2,2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data", train = False, download= True, transform = torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size = 64)




class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


tudui= Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("logs_relu")
step =0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = tudui(imgs)
    writer.add_images("output", output,step)
    step +=1

writer.close()










































