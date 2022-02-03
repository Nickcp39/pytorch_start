
import torch
if torch.cuda.is_available():
    print('it works')

import torch

torch.cuda.is_available()
print(torch.cuda.is_available())

print(dir(torch))
"这一行print torch cuda 中的独立工作项"
print(dir(torch.cuda))
"介绍 help， 查看官方解释文档 功能对于 cuda的作用， 可以发现return CUDA 可以使用"
help(torch.cuda.is_available)

from torch.utils.data import Dataset
help( Dataset)
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self):

    def __getitem__(self, idx):

