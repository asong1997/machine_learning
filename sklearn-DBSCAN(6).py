"""
DBSCAN算法思想:
    1.指定合适的𝜀和Minpoints。分别对应参数：eps= 0.2和min_samples=50
    2.计算所有的样本点，如果点p的𝜀邻域里有超过Minpoints个点，则创建一个以p为核心点的新族。
    3.反复寻找这些核心点直接密度可达（之后可能是密度可达）的点，将其加入到相应的簇，对于核心点发生“密度相连”状况的簇，给予合并。
    4.当没有新的点可以被添加到任何簇时，算法结束。

缺点：
    1.当数据量增大时，要求较大的内存支持I/O消耗也很大。
    2.当空间聚类的密度不均匀、聚类间距差相差很大时，聚类质量较差。
DBSCAN和K-MEANS比较：
    1.DBSCAN不需要输入聚类个数。
    2.聚类簇的形状没有要求。
    3.可以在需要时输入过滤噪声的参数

可视化网站：https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

x1, y1 = datasets.make_circles(n_samples=2000, factor=0.5, noise=0.05)
x2, y2 = datasets.make_blobs(n_samples=1000, centers=[[1.2, 1.2]], cluster_std=[[.1]])
x = np.concatenate((x1, x2))
plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.title("raw-data")
plt.show()


y_pred = KMeans(n_clusters=3).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("K-Means")
plt.show()

y_pred = DBSCAN().fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("DBSCAN: Default Parameters")
plt.show()

y_pred = DBSCAN(eps=0.2).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("DBSCAN: eps=0.2")
plt.show()

y_pred = DBSCAN(eps=0.2, min_samples=50).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("DBSCAN: eps=0.2, min_samples=50")
plt.show()
