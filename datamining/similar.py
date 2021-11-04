import numpy as np
from scipy import signal
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist

# 计算两个向量之间的欧式距离
def jcc_dis(a, b):
    # d = np.sqrt(np.sum(np.square(a - b)))
    X = np.vstack([a, b])
    d = pdist(X, 'jaccard')[0]
    return d


# 计算两个向量之间的余弦相似度
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos


# 对计算后的分数进行归一化，便于后续的对评价指标的加权
def normal(conf):
    conflist = np.concatenate(conf)
    mi = min(conflist)
    ma = max(conflist)
    conf = np.array(conf)
    return (conf - mi) / (ma - mi)

def showfig(sdest,epoch,s1,tile):
    show = sdest[0][0]
    show_list0 = s1[int(show[0]) * epoch:(int(show[0]) + 1) * epoch]
    show_list1 = s1[int(show[1]) * epoch:(int(show[1]) + 1) * epoch]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    plt.plot(show_list0, color='r')
    ax1.set_title('Signal 1 %s'%(tile))
    plt.ylabel('origin signal')

    ax2 = fig1.add_subplot(212)
    plt.plot(show_list1, color='r')
    ax2.set_title('Signal 2 %s'%(tile))
    plt.ylabel('origin signal')
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(r'20151026_%d_%s.png' % (e,tile))
    plt.close('all')



def similar(e):
    data = open(r'E:/Git/datamin/20151026_%d' % (e)).read()
    data = data.split()
    data = [float(s) for s in data]
    s1 = data[0:45000 * 4:4]
    lenth = len(s1)

    # 滤波
    fs = 3000
    lowcut = 1
    highcut = 30
    order = 2
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    s1_filter1 = signal.lfilter(b, a, s1)

    # 度量
    n = 200  # 设置截取频率窗的长度
    epoch = int(45000 / n)  # 计算截取出窗的个数
    cut_list = [s1_filter1[i * epoch:(i + 1) * epoch] for i in range(n)]
    # 根据截取出的时间窗个数进行截取
    jd, cd = [], []  # 定义新数组，用来保存计算出的距离与相似度
    for item1 in cut_list:
        jd.append([jcc_dis(item1, item2) for item2 in cut_list])  # 计算欧式距离
        cd.append([cos_sim(item1, item2) for item2 in cut_list])  # 计算余弦相似度
    jd_sort = normal(jd)  # 对欧式距离进行归一化
    cd_sort = 1 - normal(cd)  # 对余弦相似度进行归一化，并转换为距离
    jcd = (1.0*jd_sort + 3.5*cd_sort) / 5.0  # 对欧氏距离和余弦相似度进行加权
    jcd_sort = sorted(np.concatenate(jcd))  # 对最终得分进行排序
    # 得到最相似与最相异的波段，并去重
    jcd_same, jcd_diff = jcd_sort[n:n + 6], jcd_sort[-6:]
    samest = [np.where(jcd == jcd_same[i]) for i in range(0, 6, 2)]
    diffest = [np.where(jcd == jcd_diff[i]) for i in range(0, 6, 2)]
    # 输出最相似波段位置
    print('-------- Sam --------')
    for item,i in zip(samest,range(0,6,2)):
        print(item[0] * epoch,jcd_same[i] )

    print('-------- Dir --------')
    for item,i in zip(diffest ,range(0,6,2)):
        print(item[0] * epoch, jcd_diff[i])

    # 对最相似的波段进行可视化
    showfig(samest,epoch,s1,"same")
    showfig(diffest,epoch,s1,"differ")

if __name__ == '__main__':
    start = 113  # 从start开始做N个文件的图                    #设立变量start，作为循环读取文件的起始
    N = 2  # 设立变量N，作为循环读取文件的增量
    for e in range(start, start + N):  # 循环2次，读取113&114两个文件
        similar(e)