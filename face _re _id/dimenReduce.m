

%{
����PCA���н�ά

���룺 �����ռ� һ������ ��������������ѵ������ 
            ѵ������������ ������������
����� ����ά���ѵ�������ռ�Ͳ��������ռ�
%}

function[train, test] = dimenReduce(date, m, n)
train = []; test = [];
N=2;
%����ѵ���������������
for i = 1:N
    train = [train date(:,(m+n)*(i-1)+1:(m+n)*(i-1)+m)];
end
for i = 1:N
    test = [test date(:, (m+n)*(i-1)+1+m:(m+n)*(i-1)+m+n)];
end

clear i;

%����train�����ľ�ֵ������pca��ά����test������άǰӦ��ȥtrain�����ľ�ֵ
%ÿ��һ��������ÿ��һ������
avg = mean(train');
test = test - repmat(avg', 1, size(test,2));

%����PCA��ά
%coeff--������  score--�任������  latent--����ֵ
[coeff, score, latent] = pca(train');

%������
contrirate = cumsum(latent)./sum(latent);

%��ȡ�����ʴ���0.98�Ļ�����
n = find(contrirate>0.98, 1);
%p�д洢��ѵ������������
p = coeff(: , 1:n);

%�ֱ��ѵ�������Ͳ���������ά
test= p'*test;
train = score(:, 1:n);
train = train';

clear coeff contrirate i  score n c latent l face p;
