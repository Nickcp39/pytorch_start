
import torch
if torch.cuda.is_available():
    print('it works')

import torch

torch.cuda.is_available()
print(torch.cuda.is_available())

print(dir(torch))
#这一行print torch cuda 中的独立工作项"
print(dir(torch.cuda))
#介绍 help， 查看官方解释文档 功能对于 cuda的作用， 可以发现return CUDA 可以使用"
help(torch.cuda.is_available)

from torch.utils.data import Dataset
help( Dataset)
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  #拿到图片中所有的地址

    def __getitem__(self, idx):
        img_name = self.image[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir

        return img, label

    def __len__(self):
        return (len(self.img_path))



dir_path = "/dataset/hymenoptera_data/hymenoptera_data/train/ants"
img_path_list = os.listdir(dir_path)

# root_dir =  "C:\\Users\\nickc\PycharmProjects\\buffaloTestProject\\dataset\\hymenoptera_data\\hymenoptera_data\\train"

root_dir = "../dataset/hymenoptera_data/hymenoptera_data/train"


ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset


img, label  =  ants_dataset[11]
img.shows()
ants_dataset[123].img.shows()
