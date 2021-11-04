import numpy as np
from tsfresh import extract_features
import pandas as pd
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from pandas.plotting import  table
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']#显示中文字体


def extract(e):
    # -------- 提取数据 --------
    data = open(r'E:/Git/datamin/20151026_%d'% (e)).read()     #设立data列表变量，python 文件流，%d处，十进制替换为e值，.read读文件
    data = data.split( )                                  #以空格为分隔符，返回数值列表data
    data = [float(s) for s in data]                       #将列表data中的数值强制转换为float类型
    s1 = data[0:45000*4:4]                      #s1即为取到第一列数据，45000个数据共30秒
    # -------- 创建DataFrame对象 --------
    arr= np.array(s1)
    datalist = np.array_split(arr,30)
    timelist = {'id':[i for i in range(1,31) for j in range(1,len(datalist[0])+1)],
                'time':[j for i in range(1,31) for j in range(1,len(datalist[0])+1)],
                'data1':[j for j in arr]}
    timelist = pd.DataFrame(timelist)
    # print([timelist[timelist['id' ]== 1]])
    # -------- 设置所选特征 --------
    fc_parameters = {
        "partial_autocorrelation": [{"lag": lag} for lag in range(10)],                                        #偏自相关
        "fft_aggregated": [
            {"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]
        ],                                                                                                                              #fft聚合
        "approximate_entropy": [
            {"m": 2, "r": r} for r in [0.1, 0.3, 0.5, 0.7, 0.9]
        ],                                                                                                                             #近似熵
        "augmented_dickey_fuller": [
            {"attr": "teststat"},
            {"attr": "pvalue"},
            {"attr": "usedlag"},
        ],                                                                                                                            #ADF检验
        "fourier_entropy": [{"bins": x} for x in [2, 3, 5, 10, 100]],                                           #傅里叶熵
        "permutation_entropy": [
            {"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]                                                     #排列熵
        ],
    }
    features = extract_features(timelist, column_id="id", column_sort="time",default_fc_parameters=fc_parameters)
    print(features)
    # impute(features)
    # features_filtered = select_features(features,y, fdr_level=0.2)
    # print(features_filtered.shape)
    # fig = plt.figure(figsize=(3, 4), dpi=1400)  # dpi表示清晰度
    # ax = fig.add_subplot(111, frame_on=False)
    # ax.xaxis.set_visible(False)  # hide the x axis
    # ax.yaxis.set_visible(False)  # hide the y axis
    #
    # table(ax, features, loc='center')  # 将df换成需要保存的dataframe即可
    #
    # plt.savefig('%d_feature.jpg'%(e))
    features.to_csv('feature_%d.csv'%(e))


if __name__ == '__main__':
    start = 113 #从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
    N = 1                                                      #设立变量N，作为循环读取文件的增量
    for e in range(start,start+N):                            #循环2次，读取113&114两个文件
        extract(e)
