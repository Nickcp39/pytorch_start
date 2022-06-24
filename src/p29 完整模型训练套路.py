# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念

import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

# 训练数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                      download=True)
# 测试数据集
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数
# 损失函数中的参数可以设置， 这次可以先用默认值
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 0.01
#learning_rate = 1e-2
# 随机梯度下降
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    tudui.train() #不是必须的，但是这个叫训练模式
    for data in train_dataloader:
        # input, 真实结果 #
        imgs, targets = data
        # 预测输出
        outputs = tudui(imgs)
        # 找到预测和 真实结果之间的  loss
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        # 优化器梯度清零
        optimizer.zero_grad()
        # 反向传播， 拿到参数节点梯度
        loss.backward()
        # 每个参数节点的优化
        optimizer.step()
        # 训练次数
        total_train_step = total_train_step + 1

        # 然后就可以output 训练次数 和 loss的当前状态
        if total_train_step % 100 == 0:
            # 整100 print一次， 不要每次都print
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval() # 这里也不是必须的， 但是这个给tudui这个网络设置了一个eval 的模式， 就是 检测 testing 模式。
    total_test_loss = 0
    total_accuracy = 0
    # 没有梯度的代码， 不会进行调优
    # 可以写一些测试代码
    with torch.no_grad(): #让网络当中梯度不要了， 在这个网络中， 不需要调整梯度
        # 将测试数据集全部拿到
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            #直接算出总的误差值
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            # 整体的正确率测试
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # 记录测试次数 和每次的结果
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))  # 保存模型 每次名字 +1 
    print("模型已保存")

writer.close()


#  tensorboard --logdir=logs_train
# 用于在terminal 来看





















