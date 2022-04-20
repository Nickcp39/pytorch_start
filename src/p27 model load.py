# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
# from model_save import *
# 方式1-》保存方式1，加载模型
import torchvision
from torchvision import models
from torch import nn

model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
model = torch.load("vgg16_method2.pth")
print(vgg16)

# 陷阱1
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
tudui = Tudui() # 这一行是不需要的， 在这个陷阱中， 因为已经定义并且保存了作为pth文件，
# 可以直接调用tudui pth文件，不需要从新定义， 不需要从新定义 network的内部文件
#一般都是从新定义的， 因为这样可以保护network的纯净性， 相当于cpp文件的header文件。
model = torch.load('tudui_method1.pth')
print(model)