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

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']#显示中文字体
rcParams['figure.figsize'] = 10,8

class node(object):
    def __init__(self,left = None,right = None , leftlist = [],rightlist  = []):
        self.value = 0
        self.leftlist = leftlist
        self.rightlist = rightlist
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

def Gini(N1,N2):
    N1dic ,N2dic= {},{}
    for i in range(3):
        if i in N1:
            N1dic[i] = Counter(N1)[i]
        else:
            N1dic[i] = 0
        if i in N2:
            N2dic[i] = Counter(N2)[i]
        else:
            N2dic[i] = 0
    # pdb.set_trace()
    N1p2 = [i**2 for i in N1dic.values()]
    N2p2 = [i ** 2 for i in N2dic.values()]
    # c_num = [N1dic[i]+N2dic[i] for i in range(3)]
    if sum(N1dic.values())==0 or sum(N2dic.values())==0:
        return 0
    g1 = 1 - sum(N1p2)/pow(sum(N1dic.values()),2)
    g2 = 1 - sum(N2p2)/pow(sum(N2dic.values()),2)
    gini = sum(N1dic.values())/(sum(N1dic.values())+sum(N2dic.values()))*g1+sum(N2dic.values())/(sum(N1dic.values())+sum(N2dic.values()))*g2
    return gini

def predict(thre,sortindex,data,labels):
    N1,N2 = [],[]
    for i in sortindex:
        if data[i] > thre:
            try:
                N1.append(labels[i])
            except(IndexError):
                print(labels)
        else:
            N2.append(labels[i])
    # pdb.set_trace()
    gini = Gini(N1,N2)

    return gini

def gethre(data,labels):
    if len(data)<2:
        return data
    sortindex = np.argsort(data)
    thres = [(data[sortindex[i]]+data[sortindex[i+1]])/2.0 for i in range(len(sortindex)-1)]
    # pdb.set_trace()
    ginisplit = []
    for thre in thres:
        gini = predict(thre,sortindex,data,labels)
        ginisplit.append(gini)
    ginisort = np.argsort(ginisplit)
    # pdb.set_trace()
    try:
        return thres[ginisort[0]]
    except(IndexError):
        print(data,labels)

def nodethre(trainsize,treedata,treelabel):
    """treedata是所有的train data batch"""
    threbest = 0
    for i in range(trainsize):
        thre = gethre(treedata[i],treelabel[i])
        threbest = threbest + thre * i / trainsize  # 随着batch逐渐更新阈值thre
    return threbest

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

def nodedeci(trainsize,level,thre,nodedata,labels,batchsize):
    N1data,N2data = [[], []], [[], []]
    batch_size = batchsize
    if len(nodedata) == 0:
        return None,None
    for i in range(len(nodedata)):
        try:
            if nodedata[i]>thre:
                # for j in nodedata[0]:
                #     data.append(j[i])
                # for k in range(6):
                #     N1data[0][k].append(data[k])
                N1data[0].append(i)
                N1data[1].append(int(labels[i]))
            else:
                N2data[0].append(i)
                N2data[1].append(int(labels[i]))
        except(TypeError):
            print(nodedata[i],thre)
    return N1data,N2data

def idxdata(idxs,level,data):
    try:
        outdata = [data[0][level][i] for i in idxs]
        return outdata
    except(IndexError):
        print(idxs)


def buildtree(nodenum,trainsize,level,data,label,trainbatch,idx,batchsize):
    # print('================')
    # print(level,len(treedata))
    # print('=============')
    # print(treelabel)
    Node = node()
    nodenum += 1
    print(nodenum)
    # print(trainbatch)
    # print(trainbatch[0][level], trainbatch[1])
    # print(trainbatch[0][level+1])
    # thre = nodethre(trainsize, nodedata,nodelabel)
    if data is None:
        return None,nodenum
    thre = gethre(data,label)
    print("===thre:",thre)
    print(data)
    N1idx,N2idx = nodedeci(trainsize,level,thre,data,label,batchsize)
    if N1idx is None or N2idx is None:
        return None,nodenum
    N1, N2= N1idx[1],N2idx[1]
    # if test[level] > thre:
    #     N1 = N1idx[1]+['T']
    #     cou = Counter(N1idx[1])
    #     lab = [i for i in cou.keys() if max(cou.values()) == cou[i]]
    # else:
    #     N2 = N2idx[1]+['T']
    #     cou = Counter(N2idx[1])
    #     lab = [i for i in cou.keys() if max(cou.values()) == cou[i]]

    print("========N1,N2=====", N1, N2)
    # N1data = idxdata(N1idx[0], level+1, trainbatch)
    # print("-----------",N1data)
    Node.value = thre
    Node.leftlist = N1
    Node.rightlist = N2
    # if  len(N1idx[0]) >0 or len(N2idx[0]) >0:
        # print(len(N1data),N1data)

    if len(N1idx) > 0 and len(Counter(N1idx[1]).keys()) >1:
        level = level + 1
        if level < 6:
            print('++++left+++++')
            N1data = idxdata(N1idx[0], level, trainbatch)
            leftnode,nodenum= buildtree(nodenum,trainsize,level,N1data,N1idx[1],trainbatch,N1idx[0],batchsize)
            Node.left = leftnode
    if nodenum > 5 :
        return None,nodenum
    if len(N2idx) > 0 and len(Counter(N2idx[1]).keys()) >1:
        level = level + 1
        if level < 6:
            print('++++Right+++++')
            N2data = idxdata(N2idx[0], level, trainbatch)
            rightnode,nodenum= buildtree(nodenum,trainsize, level, N2data,N2idx[1],trainbatch,N2idx[0],batchsize)
            Node.right = rightnode

    return Node,nodenum

def trastree(head,newtest,level,flag,prelab):
    if head != None:
        if newtest[level] > head.value:
            head.leftlist.append('T')
        else:
            head.rightlist.append('T')
        print('------L,R------',head.leftlist,head.rightlist)
        # if head.left is None and head.right is None:
        #     print('叶子')
        #     if 'T' in head.leftlist:
        #         cou = Counter(head.leftlist)
        #         lab = [i for i in cou.keys() if max(cou.values()) == cou[i]]
        #         if flag == 0:
        #             prelab = lab[0]
        #             print('predict', prelab)
        #             flag  = 1
        #     elif 'T' in head.rightlist:
        #         cou = Counter(head.rightlist)
        #         lab = [i for i in cou.keys() if max(cou.values()) == cou[i]]
        #         if flag == 0:
        #             prelab = lab[0]
        #             print('predict', prelab)
        #             flag = 1
        if head.left is not None and 'T' in head.leftlist:
            level += 1
            L = trastree(head.left,newtest,level,flag,prelab)
        if head.right is not None and  'T' in head.rightlist:
            level += 1
            R = trastree(head.right, newtest, level,flag,prelab)
        return prelab

def main(e):
    # features = extract(e)
    count = 0
    trainsize = 20
    level = 0
    batchidx = 5

    batchsize = 10
    nodenum = 0
    newtest = []
    data,keylist = getdata(e)
    datalabel = labeled(data,keylist)
    sharpdata = sharpdataset(dataset=data,datalabel=datalabel,keylist=keylist)
    dataloader = DataLoader(dataset=sharpdata,batch_size=batchsize,shuffle=False)
    traindata, testdata = splitdata(dataloader,trainsize)
    # treedatas = [batch[0][level] for batch in traindata] #第level个特征属性的20个batch
    # treelabels = [batch[1] for batch in traindata]
    testidx = 0
    trainbatch = traindata[batchidx]
    data = trainbatch[0][level]
    labelt = trainbatch[1]
    print(labelt)
    testdata1 = testdata[0][0]
    testlabel = [int(i) for i in testdata[0][1]]
    for i in range(10):
        newtest.append([testdata1[idx][i] for idx in range(6)])
    for testidx in range(1):
        tree,num= buildtree(nodenum,trainsize,level,data,labelt,trainbatch,trainbatch[2],batchsize)
        lab = trastree(tree, newtest[testidx], 0,0,None)
        print(newtest[testidx])
        print('GT',testlabel[testidx])
        print('prelabel',lab)
        if testlabel[testidx] == lab:
            count += 1
    allacc = count / 10.0
    print("ACC = ", round(allacc, 2))





if __name__ == '__main__':
    start = 113 #从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
    N = 1                                                      #设立变量N，作为循环读取文件的增量
    for e in range(start,start+N):                            #循环2次，读取113&114两个文件
        main(e)
