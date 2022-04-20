

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset_transform =  torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()

])



train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform= dataset_transform, download = True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform= dataset_transform,download= True)



# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0] #查看数据 label标记
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0]) # 用了transform 转换 totensor之后， 就可以用tensor数据类型来处理原始图片。


writer = SummaryWriter("../p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set",img,i )

writer.close() # 把读写进行关闭





























