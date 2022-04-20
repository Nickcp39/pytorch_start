


from torch.utils.tensorboard import SummaryWriter  #导入tensorboard 导入常规类

import numpy as np     #是为了读取图片用numpy型
from PIL import Image

writer = SummaryWriter("../logs")  # 创建一些实例

image_path = "../dataset/train/ants_image/6240329_72c01e663e.jpg"
image_path = 'dataset/train/ants_image/0013035.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 3, dataformats='HWC')   # 添加方法， 加一个img_array =tag， 加一个 global =1 ， 加一个dataformat = “HWC"`
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i) # 标记数字

writer.close()

img_PIL = Image.open(image_path)






