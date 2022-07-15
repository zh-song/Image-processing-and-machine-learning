

%{
利用PCA进行降维

输入： 样本空间 一个矩阵 包含测试样本和训练样本 
            训练样本个数， 测试样本个数
输出： 经降维后的训练样本空间和测试样本空间
%}

function[train, test] = pcad(date, input)
train = []; test = [];

%划分训练样本与测试样本

train = [date];

% for i = 1:N
%     test = [test input];
% end
test = [input];
clear i;

%先求train样本的均值，再用pca降维，对test样本降维前应减去train样本的均值
%每行一个样本，每列一个特征
avg = mean(train');
test = test - repmat(avg', 1, size(test,2));

%利用PCA降维
%coeff--基向量  score--变换后样本  latent--特征值
[coeff, score, latent] = pca(train');

%贡献率
contrirate = cumsum(latent)./sum(latent);

%提取贡献率大于0.98的基向量
n = find(contrirate>0.98, 1);
%p中存储各训练样本基向量
p = coeff(: , 1:n);

%分别对训练样本和测试样本降维
test= p'*test;
train = score(:, 1:n);
train = train';

clear coeff contrirate i  score n c latent l face p;
