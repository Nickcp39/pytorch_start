

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#准备测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train = False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size = 16, shuffle= True, num_workers = 0, drop_last = False )
# 测试数据集 中的 第一张图片  样本
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("../dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()










































