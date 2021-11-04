import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import paired_distances
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

def getdata(e):
    data = pd.read_excel(r'E:/Git/datamin/feature_%d.xlsx'% (e),sheet_name=0)
    keylist = data.keys()[1:]
    # print(keylist)

    return data,keylist

def getitem(data,__factory,idx):
    features = []
    for i in range(6):
        features.append(data[__factory[i]][idx])
    return features

def viewdata(data,__factory):
    tradata = []
    for idx in range(300):
        tradata.append(getitem(data,__factory,idx))
    return tradata

def caldis(x,y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(sum((x-y)**2))

def showk(klst,SSElst):
    plt.plot(klst,SSElst)
    plt.show()

def get_closest_dist(data,idx, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i in centroids:
        dist = caldis(data[i], data[idx])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def init(data,num,k):
    cluster_centers= []
    cluster_centers.append(random.randint(0,num))
    d = [0 for _ in range(num)]
    for _ in range(1, k):
        total = 0.0
        for i in range(num):
            d[i] = get_closest_dist(data,i, cluster_centers)  # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d):  # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(i)
            break
    return cluster_centers


class Kmeans(object):
    def __init__(self,data = None,k = 3,factory = None,num = 300,randominit = False):
        self.k = k
        self.datas = data
        self.labels = [-1]*num
        self.fac = factory
        if randominit:
            self.cenidx = random.sample(list(range(num)),k)
        else:
            self.cenidx = init(data,num,k)
        self.centers = [self.datas[i] for i in self.cenidx]
        for i in range(0,k):
            self.labels[self.cenidx[i]] = i

    def forward(self,num=300):
        eps = [100]*self.k
        last_label = [self.k+1]*num
        while last_label != self.labels:
            last_label = self.labels
            for i in range(num):
                mind = 100
                for j in range(self.k):
                    diss = caldis(self.datas[i],self.centers[j])
                    if diss < mind:
                        mind = diss
                        self.labels[i] = j

            for m in range(self.k):
                lst = [self.datas[i] for i in self.labels if i == m]
                self.centers[m] = np.mean(lst,axis=0)
                # pdb.set_trace()
                epsc = np.var(lst,axis=0)
                eps[m] = np.mean(epsc)

        return self.labels,np.mean(eps)


def main(file):
    __factory = {
        0: 'data1__partial_autocorrelation__lag_5',
        1: 'data1__fft_aggregated__aggtype_"centroid"',
        2: 'data1__approximate_entropy__m_2__r_0.3',
        3: 'data1__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"',
        4: 'data1__fourier_entropy__bins_10',
        5: 'data1__permutation_entropy__dimension_3__tau_1',
    }
    SSElst , minSSE = [] , float('inf')
    klst = list(range(1,11))
    rawdata,keylist = getdata(file)
    data = viewdata(rawdata,__factory)
    for k in klst:
        kmeans = Kmeans(data,k,__factory,num=300)
        label,SSE = kmeans.forward(num=300)
        # if SSE < minSSE and SSE > 0:
        #     minSSE = SSE
        #     bestlabel = label
        #     bestk = k
        SSElst.append(SSE)
    # print(bestk)
    showk(klst,SSElst)
    # k = 6
    # kmeans = Kmeans(data, k, __factory, num=300)
    # bestlabel, SSE = kmeans.forward(num=300)
    # # # print(label)
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(rawdata[__factory[1]],rawdata[__factory[2]],rawdata[__factory[4]],c=bestlabel)  # 绘制聚类结果
    plt.show()



if __name__ == '__main__':
    main(113)