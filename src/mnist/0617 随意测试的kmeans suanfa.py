import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from skimage import data, io
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
dog = imread('vesseltest.PNG')
plt.figure(num=None, figsize=(8, 6), dpi=80)
imshow(dog);
#io.imshow(dog)
#plt.show()

"""
这使得图像的操作更加简单，
因为更容易将其视为可以输入到机器学习算法中的数据。

在我们的实例中，我们将使用K-means算法对图像进行聚类。
"""
def image_to_pandas(image):
    df = pd.DataFrame([image[:,:,0].flatten(),
                       image[:,:,1].flatten(),
                       image[:,:,2].flatten()
                       ]).T
    df.columns = ["Red_Channel","Green_Channel","Blue_Channel"]
    return df
df_doggo = image_to_pandas(dog)
df_doggo.head(5)

"""
如我们所见，图像被分为四个不同的区域。让我们分别可视化每个区域。
"""

plt.figure(num=None, figsize=(8, 6), dpi=80)
kmeans = KMeans(n_clusters=  100,   random_state = 10).fit(df_doggo)
result = kmeans.labels_.reshape(dog.shape[0],dog.shape[1])
imshow(result, cmap='viridis')
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for n, ax in enumerate(axes.flatten()):
    ax.imshow(result == [n], cmap='gray');
    ax.set_axis_off()

fig.tight_layout()
plt.show()

"""
这使得图像的操作更加简单，因为更容易将其视为可以输入到机器学习算法中的数据。

在我们的实例中，我们将使用K-means算法对图像进行聚类。
"""

fig, axes = plt.subplots(1,3, figsize=(15, 12))
for n, ax in enumerate(axes.flatten()):
    dog = imread('beach_doggo.png')
    dog[:, :, 0] = dog[:, :, 0]*(result==[n])
    dog[:, :, 1] = dog[:, :, 1]*(result==[n])
    dog[:, :, 2] = dog[:, :, 2]*(result==[n])
    ax.imshow(dog);
    ax.set_axis_off()

fig.tight_layout()

"""
我们可以看到，该算法生成了三个不同的簇，即沙子，生物和天空。
当然，算法本身对这些簇并不十分在意，只是它们共享相似的RGB值。
要由人类来解释这些簇。

在我们离开之前，我认为如果简单地将其绘制在3D图形上，
则对实际显示我们的图像会有所帮助。
"""


def pixel_plotter(df):
    x_3d = df['Red_Channel']
    y_3d = df['Green_Channel']
    z_3d = df['Blue_Channel']

    color_list = list(zip(df['Red_Channel'].to_list(),
                          df['Blue_Channel'].to_list(),
                          df['Green_Channel'].to_list()))
    norm = colors.Normalize(vmin=0, vmax=1.)
    norm.autoscale(color_list)
    p_color = norm(color_list).tolist()

    fig = plt.figure(figsize=(12, 10))
    ax_3d = plt.axes(projection='3d')
    ax_3d.scatter3D(xs=x_3d, ys=y_3d, zs=z_3d,
                    c=p_color, alpha=0.55);

    ax_3d.set_xlim3d(0, x_3d.max())
    ax_3d.set_ylim3d(0, y_3d.max())
    ax_3d.set_zlim3d(0, z_3d.max())
    ax_3d.invert_zaxis()

    ax_3d.view_init(-165, 60)


pixel_plotter(df_doggo)

"""
我们应该记住，这实际上是算法定义“紧密度”的方式。
如果我们将K-Means算法应用于该图，则其分割图像的方式将变得非常清晰。
"""

df_doggo['cluster'] = result.flatten()
plt.show()

def pixel_plotter_clusters(df):
    x_3d = df['Red_Channel']


    y_3d = df['Green_Channel']
    z_3d = df['Blue_Channel']

    fig = plt.figure(figsize=(12, 10))
    ax_3d = plt.axes(projection='3d')
    ax_3d.scatter3D(xs=x_3d, ys=y_3d, zs=z_3d,
                    c=df['cluster'], alpha=0.55);

    ax_3d.set_xlim3d(0, x_3d.max())
    ax_3d.set_ylim3d(0, y_3d.max())
    ax_3d.set_zlim3d(0, z_3d.max())
    ax_3d.invert_zaxis()

    ax_3d.view_init(-165, 60)


pixel_plotter_clusters(df_doggo)
plt.show()


"""
K-Means算法是一种流行的无监督学习算法，任何数据科学家都可以轻松使用它。虽然它很简单，但对于像素差异非常明显的图像来说，它的功能非常强大。
在以后的文章中，我们将介绍其他机器学习算法，
我们可以用于图像分割以及微调超参数。
但现在，我希望你可以在自己的任务中使用这种方法。
"""










