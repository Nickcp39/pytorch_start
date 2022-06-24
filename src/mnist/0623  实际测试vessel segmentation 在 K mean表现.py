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
from PIL import Image
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets

sns.set()

dog = imread('vesseltest_ground_truth.PNG')
plt.figure(num=None, figsize=(8, 6), dpi=80)

imshow(dog);
#io.imshow(dog)
#plt.show()

"""
这使得图像的操作更加简单，
因为更容易将其视为可以输入到机器学习算法中的数据。

在我们的实例中，我们将使用K-means算法对图像进行聚类。
"""
imframe = Image.open('vesseltest_ground_truth.PNG')
npframe = np.array(imframe.getdata())
imgrgbdf = pd.DataFrame(npframe)

df_doggo = imgrgbdf
df_doggo.head(5)

"""
kmeans = KMeans(n_clusters= 100, random_state = 10).fit(df_doggo)
result = kmeans.labels_.reshape(dog.shape[0],dog.shape[1])
imshow(result, cmap='viridis')
plt.show()
"""




DBSCAN_result = DBSCAN().fit_predict(df_doggo)
imshow(DBSCAN_result)
plt.show()

# -*- coding:utf-8 -*-
"""
Description: DBSCAN简易版实现，主要sklearn实现内存占用过高

@author: WangLeAi
@date: 2018/12/25
"""

import numpy as np

UNCLASSIFIED = False
NOISE = -1


def __dis(vector1, vector2):
    """
    余弦夹角距离
    :param vector1: 向量A
    :param vector2: 向量B
    :return:
    """
    distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
    distance = max(0.0, 1.0 - float(distance))
    return distance


def __eps_neighborhood(vector1, vector2, eps):
    """
    是否邻居
    :param vector1: 向量A
    :param vector2: 向量B
    :param eps: 同一域下样本最大距离
    :return:
    """
    return __dis(vector1, vector2) < eps


def __region_query(data, point_id, eps):
    """
    核心函数，区域查询
    :param data: 数据集,array
    :param point_id: 核心点
    :param eps: 同一域下样本最大距离
    :return:
    """
    n_points = data.shape[0]
    seeds = []
    for i in range(0, n_points):
        if __eps_neighborhood(data[point_id, :], data[i, :], eps):
            seeds.append(i)
    return seeds


def __expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
    """
    类簇扩散
    :param data: 数据集,array
    :param classifications: 分类结果
    :param point_id: 当前点
    :param cluster_id: 分类类别
    :param eps: 同一域下样本最大距离
    :param min_points: 每个簇最小核心点数
    :return:
    """
    seeds = __region_query(data, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        mark = False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
        while len(seeds) > 0:
            current_point = seeds[0]
            results = __region_query(data, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        mark = True
    return mark


def dbscan(data, eps, min_points):
    """
    dbscan聚类
    :param data: 数据集,array
    :param eps: 同一域下样本最大距离
    :param min_points: 每个簇最小核心点数
    :return:
    """
    cluster_id = 1
    n_points = data.shape[0]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if __expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications


# if __name__ == "__main__":
    # eps为距离阈值ϵ，min_samples为邻域样本数阈值MinPts,X为数据


