"""
代码目标：基于python实现K-Means算法
实现思路：
    1.先从没有标签的元素集合A中随机取k个元素，作为k个子集各自的重心。
    2.分别计算剩下的元素到k个子集重心的距离（这里的距离也可以使用欧氏距离），根据距离将这些元素分别划归到最近的子集。
    3.根据聚类结果，重新计算重心（重心的计算方法是计算子集中所有元素各个维度的算数平均数）。
    4.将集合A中全部元素按照新的重心然后再重新聚类。
    5.重复第4步，直到聚类结果不再发生变化
缺点：
    1.计算量大，收敛时间长                         解决办法：Mini Batch K-Means（随机选取数据子集聚类）
    2.对k个初始质心的选择比较敏感，容易陷入局部最小值。  解决办法：多次随机初始化，取最优结果
    3.K值不好选取。                               解决办法：肘部法、轮廓系数
    4.对于”非球状数据“聚类效果比较差                  解决办法：基于密度聚类（DBSCAN、Mean Shift）
"""

import numpy as np
import matplotlib.pyplot as plt


def raw_data_plot():
    # 载入数据
    plt.figure(figsize=(6, 4))
    data = np.genfromtxt("./data/kmeans.txt", delimiter=" ")
    plt.scatter(data[:, 0], data[:, 1])
    plt.title("Scatter diagram of data distribution")
    plt.savefig("./img/scatter.png")
    # plt.show()


# 计算距离
def euclDistance(vector1, vector2):
    return np.sqrt(sum((vector2 - vector1) ** 2))


# 初始化质心
def initCentroids(data, k):
    numSamples, dim = data.shape
    # k个质心，列数跟样本的列数一样
    centroids = np.zeros((k, dim))
    # 随机选出k个质心
    for i in range(k):
        # 随机选取一个样本的索引
        index = int(np.random.uniform(0, numSamples))
        # 作为初始化的质心
        centroids[i, :] = data[index, :]
    return centroids


# 传入数据集和k的值
def kmeans(data, k):
    # 计算样本个数
    numSamples = data.shape[0]
    # 样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
    clusterData = np.array(np.zeros((numSamples, 2)))
    # 决定质心是否要改变的变量
    clusterChanged = True

    # 初始化质心
    centroids = initCentroids(data, k)

    while clusterChanged:
        clusterChanged = False
        # 循环每一个样本
        for i in range(numSamples):
            # 最小距离
            minDist = 100000.0
            # 定义样本所属的簇
            minIndex = 0
            # 循环计算每一个质心与该样本的距离
            for j in range(k):
                # 循环每一个质心和样本，计算距离
                distance = euclDistance(centroids[j, :], data[i, :])
                # 如果计算的距离小于最小距离，则更新最小距离
                if distance < minDist:
                    minDist = distance
                    # 更新最小距离
                    clusterData[i, 1] = minDist
                    # 更新样本所属的簇
                    minIndex = j

                    # 如果样本的所属的簇发生了变化
            if clusterData[i, 0] != minIndex:
                # 质心要重新计算
                clusterChanged = True
                # 更新样本的簇
                clusterData[i, 0] = minIndex

        # 更新质心
        for j in range(k):
            # 获取第j个簇所有的样本所在的索引
            cluster_index = np.nonzero(clusterData[:, 0] == j)
            # 第j个簇所有的样本点
            pointsInCluster = data[cluster_index]
            # 计算质心
            centroids[j, :] = np.mean(pointsInCluster, axis=0)
        #         showCluster(data, k, centroids, clusterData)
    return centroids, clusterData


# 显示结果
def showCluster(data, k, centroids, clusterData):
    numSamples, dim = data.shape
    if dim != 2:
        print("dimension of your data is not 2!")
        return 1
    # 用不同颜色形状来表示各个类别
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Your k is too large!")
        return 1
    plt.figure(figsize=(6, 4))
    # 画样本点
    for i in range(numSamples):
        markIndex = int(clusterData[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[markIndex])
        # 用不同颜色形状来表示各个类别
    mark = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画质心点
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=20)
    plt.title("result of cluster")
    plt.savefig("./img/cluster_result.png")
    # plt.show()


def predict(datas):
    """
        # 做预测
        x_test = [0, 1]
        np.tile(x_test, (k, 1))
        # 误差
        np.tile(x_test, (k, 1)) - centroids
        # 误差平方
        (np.tile(x_test, (k, 1)) - centroids) ** 2
        # 误差平方和
        ((np.tile(x_test, (k, 1)) - centroids) ** 2).sum(axis=1)
        # 最小值所在的索引号
        np.argmin(((np.tile(x_test, (k, 1)) - centroids) ** 2).sum(axis=1))
    """
    return np.array([np.argmin(((np.tile(data, (k, 1)) - centroids) ** 2).sum(axis=1)) for data in datas])


if __name__ == '__main__':
    # 载入原始数据
    data = np.genfromtxt("./data/kmeans.txt", delimiter=" ")
    # 数据可视化
    raw_data_plot()
    # 设置k值
    k = 4
    # centroids 簇的中心点
    # cluster Data样本的属性，第一列保存该样本属于哪个簇，第二列保存该样本跟它所属簇的误差
    centroids, clusterData = kmeans(data, k)
    if np.isnan(centroids).any():
        print('Error')
    else:
        print('cluster complete!')
    # 显示聚类结果
    showCluster(data, k, centroids, clusterData)
    # 打印聚类质心
    print(centroids)

    # 预测新样本
    tests = [[-3.156485, 3.191137],
             [3.165506, - 3.999838],
             [-2.786837, - 3.099354],
             [4.208187, 2.984927],
             [-2.123337, 2.943366],
             [0.704199, - 0.479481],
             [-0.392370, - 3.963704],
             [2.831667, 1.574018]]
    preds = predict(tests)
    print(preds)

    # 画出作用域
    # 获取数据值所在的范围
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 3
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 3
    # 生成网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    z = predict(np.c_[xx.ravel(), yy.ravel()])  # ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
    z = z.reshape(xx.shape)
    # 等高线图
    cs = plt.contourf(xx, yy, z)
    # 显示结果
    showCluster(data, k, centroids, clusterData)
    plt.title("Clustering scope")
    plt.savefig("./img/Clustering_scope.png")
    plt.show()
