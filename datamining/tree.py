import numpy as np
import csv, random

def gini(left_tt, left_ff, right_tt, right_ff):
    sum_left = left_tt+left_ff
    sum_right = right_tt+right_ff
    num = sum_left + sum_right
    n1 = 1 - (left_tt/(sum_left))**2 - (left_ff/(sum_left))**2
    n2 = 1 - (right_tt/(sum_right))**2 - (right_ff/(sum_right))**2
    return n1*(sum_left)/num + n2*(sum_right)/num

def train_tree(data, label, feature_id = 0, classleft = [1], classright = [2,3]):
    data_number = len(data) # 读入数据长度
    data = np.array(data) 
    one_feature = data[:,feature_id] # 取feature_id列特征
    sort_feature = np.sort(one_feature) # 排列数据
    gini_list = [] # gini系数的值
    iter_feature_thr = sort_feature[0] - 1
    feature_thr = [iter_feature_thr] # gini系数所对应的特征阈值

    for i in range(data_number+1):
        left_tt, left_ff, right_tt, right_ff = 0, 0, 0, 0
        for k in range(data_number): 
            if one_feature[k] < iter_feature_thr: # 小于阈值，即分到左子树中的数据
                if label[k] in classleft: # tt为当前类别，ff为其他类别
                    left_tt += 1
                else:
                    left_ff += 1
            else: # 大于阈值，即分到右子树中的数据
                if label[k] in classright: 
                    right_tt += 1
                else:
                    right_ff += 1

        # 获取Gini系数
        if (left_tt+left_ff)==0 or (right_tt+right_ff)==0:
            gini_list.append(10000)
        else:
            gini_list.append(gini(left_tt, left_ff, right_tt, right_ff))

        # 获取每次选取的特征阈值
        if i < len(sort_feature)-1:
            iter_feature_thr = (sort_feature[i]+sort_feature[i+1])/2
        elif i == len(sort_feature)-1:
            iter_feature_thr = sort_feature[i] + 1
        else:
            break
        feature_thr.append(iter_feature_thr)

    # 输出结果
    lowest_gini_id = np.argmin(gini_list) # 获取最低的gini值所对应的
    best_feature_thr = feature_thr[lowest_gini_id] 
    print('左支路个数',lowest_gini_id)
    print('右支路个数',data_number-lowest_gini_id)
    print('该节点所对应的阈值', best_feature_thr)
    # 获取分到左子树和右子树的数据，方便下次迭代
    data_left, data_right = [],[]
    for i,item in enumerate(data):
        if data[i][feature_id] < best_feature_thr:
            data_left.append(data[i])
        else:
            data_right.append(data[i])
    return np.array(data_left), np.array(data_right), best_feature_thr

def val_tree(data, best_feature_thr, feature_id = 0):
    data = np.array(data)
    data_left, data_right = [],[]
    for i,item in enumerate(data):
        if data[i][feature_id] < best_feature_thr:
            data_left.append(data[i])
        else:
            data_right.append(data[i])
    return np.array(data_left), np.array(data_right)

if __name__ == '__main__':                                                   
    # 训练
    label = [random.randint(1,3) for i in range(300)] 
    # 将文件中的数据读取到data文件中，并将其转为data
    data = [] 
    with open("feature_113.csv",'r') as f:
        txtList = csv.reader(f)
        for i,item in enumerate(txtList):
            if i == 0:
                continue
            data.append([float(st) for st in item[1:]])
    tree1_left, tree1_right, tree1_thr = train_tree(data, label, feature_id = 0, classleft = [1], classright = [2,3])
    tree2_1_left, tree2_1_right, tree2_1_thr = train_tree(tree1_left, label, feature_id = 1, classleft = [1], classright = [2,3])
    tree2_2_left, tree2_2_right, tree2_2_thr = train_tree(tree1_right, label, feature_id = 2, classleft = [1,2], classright = [3])
    tree3_1_left, tree3_1_right, tree3_1_thr = train_tree(tree2_1_left, label, feature_id = 3, classleft = [1], classright = [2])
    tree3_2_left, tree3_2_right, tree3_2_thr = train_tree(tree2_1_right, label, feature_id = 4, classleft = [2], classright = [3])
    tree3_3_left, tree3_3_right, tree3_3_thr = train_tree(tree2_2_left, label, feature_id = 5, classleft = [1], classright = [2])

    # 测试
    label_val = [random.randint(1,3) for i in range(300)] 
    # 将文件中的数据读取到data文件中，并将其转为data
    data_val = [] 
    with open("feature_114.csv",'r') as f:
        txtList = csv.reader(f)
        for i,item in enumerate(txtList):
            if i == 0:
                continue
            data_val.append([float(st) for st in item[1:]])
    tree1_left, tree1_right = val_tree(data_val, tree1_thr, feature_id = 0)
    tree2_1_left, tree2_1_right = val_tree(tree1_left, tree2_1_thr, feature_id = 1)
    tree2_2_left, tree2_2_right = val_tree(tree1_right, tree2_2_thr, feature_id = 2)
    tree3_1_left, tree3_1_right = val_tree(tree2_1_left, tree3_1_thr, feature_id = 3)
    tree3_2_left, tree3_2_right = val_tree(tree2_1_right, tree3_2_thr, feature_id = 4)
    tree3_3_left, tree3_3_right = val_tree(tree2_2_left, tree3_3_thr, feature_id = 5)


    print(tree1_left.shape,tree1_right.shape)
    print(tree2_1_left.shape, tree2_1_right.shape, tree2_2_left.shape, tree2_2_right.shape)
    print(tree3_1_left.shape, tree3_1_right.shape, tree3_2_left.shape, tree3_2_right.shape, tree3_3_left.shape, tree3_3_right.shape)

    confusion_matrix = np.zeros((3,3))
    for i, item in enumerate(data_val):
        if item in tree3_1_left or item in tree3_3_left:
            res = 1
        elif item in tree3_1_right or tree3_2_left in tree3_3_right:
            res = 2
        else:
            res = 3
        if label_val[i] == 1:
            confusion_matrix[0][res-1] += 1
        elif label_val[i] == 2:
            confusion_matrix[1][res-1] += 1
        else:
            confusion_matrix[2][res-1] += 1
    print(confusion_matrix)

    





