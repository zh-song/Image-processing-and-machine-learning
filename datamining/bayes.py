import numpy as np
import matplotlib.pyplot as plt
from tsfresh import extract_features
import pandas as pd
from sklearn.cluster import KMeans
import xlrd
from pylab import rcParams
from collections import Counter
from datatset import sharpdataset
from torch.utils.data import DataLoader
import pdb
from sklearn.metrics.pairwise import paired_distances
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']#显示中文字体
rcParams['figure.figsize'] = 10,8
from sklearn.naive_bayes import GaussianNB


class node(object):
    def __init__(self,left = None,right = None):
        self.value = 0
        self.left = left
        self.right = right

def extract(e):
    # -------- 提取数据 --------
    data = open(r'E:/Git/datamin/20151026_%d'% (e)).read()     #设立data列表变量，python 文件流，%d处，十进制替换为e值，.read读文件
    data = data.split( )                                  #以空格为分隔符，返回数值列表data
    data = [float(s) for s in data]                       #将列表data中的数值强制转换为float类型
    s1 = data[0:45000*4:4]                      #s1即为取到第一列数据，45000个数据共30秒
    # -------- 创建DataFrame对象 --------
    arr= np.array(s1)
    datalist = np.array_split(arr,300)
    timelist = {'id':[i for i in range(1,301) for j in range(1,len(datalist[0])+1)],
                'time':[j for i in range(1,301) for j in range(1,len(datalist[0])+1)],
                'data1':[j for j in arr]}
    timelist = pd.DataFrame(timelist)
    # -------- 设置所选特征 --------
    fc_parameters = {
        "partial_autocorrelation": [{"lag": lag} for lag in [5]],                                        #偏自相关
        "fft_aggregated": [
            {"aggtype": s} for s in ["centroid"]
        ],                                                                                                                              #fft聚合
        "approximate_entropy": [
            {"m": 2, "r": r} for r in [0.3]
        ],                                                                                                                             #近似熵
        "augmented_dickey_fuller": [
            {"attr": "pvalue"},
        ],                                                                                                                            #ADF检验
        "fourier_entropy": [{"bins": x} for x in [10]],                                           #傅里叶熵
        "permutation_entropy": [
            {"tau": 1, "dimension": x} for x in [3]                                                     #排列熵
        ],
    }
    features = extract_features(timelist, column_id="id", column_sort="time",default_fc_parameters=fc_parameters)
    # print(features["data1__partial_autocorrelation__lag_5"])
    features.to_excel('feature_%d.xlsx'%(e))
    return features

def getdata(e):
    data = pd.read_excel(r'E:/Git/datamin/feature_%d.xlsx'% (e),sheet_name=0)
    keylist = data.keys()[1:]
    # print(keylist)
    return data,keylist

def labeled(data,keylist):
    data_x,data_y = data["data1__permutation_entropy__dimension_3__tau_1"], data["data1__fourier_entropy__bins_10"]
    # print(data["data1__permutation_entropy__dimension_3__tau_1"][299])
    newdata= [(data_x[i],data_y[i]) for i in range(0,300)]
    # plt.scatter(data_x,data_y)  # 绘制原始数
    # plt.show()
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(newdata)
    datalable = kmeans.predict(newdata)  # 获取聚类分配
    print(Counter(datalable))
    # plt.scatter(data_x,data_y, c=datalable)  # 绘制聚类结果
    # plt.show()
    return datalable

def splitdata(dataloader,trainsize):
    """主要操作为 traindata.append(datab)

    datab[0]：特征值6列，datab[1]：labels，datab[2]：raw index"""
    traindata,testdata = [],[]
    for i,datab in enumerate(dataloader):        #30个batch，10个test
        if i < trainsize:
            traindata.append(datab)
        else:
            testdata.append(datab)
    return traindata,testdata

def KNN(testdata,traindata,labels,k):

    label = []
    dismap = [[] for i in range(10)]

    for i in range(10):
        for j in range(10):
            a = np.array(newtest[i])
            b = np.array(newtrain[j])
            a = a.reshape(1,-1)
            b = b.reshape(1,-1)
            d = paired_distances(a,b,metric="cosine")
            dismap[i].append(float(d))
    for i in range(10):
        idxs = np.argsort(dismap[i])
        nears = idxs[0:k]
        mostlabel = [int(labels[i]) for i in nears]
        cou = Counter(mostlabel)
        a = [i for i in cou.keys() if max(cou.values()) == cou[i]]
        label.append(a[0])
    return label

def main(e):
    # features = extract(e)
    newtest, newtrain = [], []
    trainsize = 20
    acc = []
    data,keylist = getdata(e)
    datalabel = labeled(data,keylist)
    sharpdata = sharpdataset(dataset=data,datalabel=datalabel,keylist=keylist)
    dataloader = DataLoader(dataset=sharpdata,batch_size=10,shuffle=False)
    traindatas, testdatas = splitdata(dataloader,trainsize)
    # treedatas = [batch[0][level] for batch in traindata] #第level个特征属性的20个batch
    # treelabels = [batch[1] for batch in traindata]
    for i in range(10):
        for batchidx in range(20):
            traindata = traindatas[batchidx][0]
            labels = traindatas[batchidx][1]
            testdata = testdatas[i][0]
            gt = testdatas[i][1]
            count = 0
            for i in range(10):
                # newtest.append([testdata[idx][i] for idx in range(6)])
                newtrain.append([traindata[idx][i] for idx in range(6)])
            print(newtrain)
            clf = GaussianNB()
            clf.fit(newtrain, labels)
            label = clf.predict(newtest)
            # print(label)
            # print(gt)
            for i in range(10):
                if label[i] == gt[i]:
                    count +=1
            acc.append(count/10.0)
    allacc = sum(acc)/len(acc)
    print("ACC = ",round(allacc,2))





if __name__ == '__main__':
    start = 113 #从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
    N = 1                                                      #设立变量N，作为循环读取文件的增量
    for e in range(start,start+N):                            #循环2次，读取113&114两个文件
        main(e)
