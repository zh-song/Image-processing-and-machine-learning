%orl人脸库
% N = 40; m = 5; n = 5;
% N: 共有40组人脸图片  m:训练样本个数  n:测试样本个数
N = 40; m = 5; n = 1;
% 每一列为一个样本 

%读取人脸信息
for i=1:N
    for j=1:m+n
        %原始人脸信息
        % face{i,j}=im2double(imread(strcat('orl_faces\s',num2str(i),'\s',num2str(i),'-',num2str(j),'.pgm')));
        face{i,j}=im2double(imread(strcat('dataset\',num2str(i),'\00',num2str(i),'0',num2str(j),'.bmp')));
         if ndims(face{i,j})==3
             face{i,j} = rgb2gray(face{i,j});
         end
         [ox, oy] = size(face{i,j});
         orlFace(:,(i-1)*(m+n)+j) = reshape(face{i,j}, ox*oy, 1);
    end
end

clear i c s cA cH cV cD ox oy dx dy;

%划分训练集测试集，用pca降维
[orlTrain, orlTest] = dimenReduce(orlFace, m, n);

%计算欧式距离，用最近邻法分类
for i=1:N*n
    for j=1:N*m
        Dis(i,j) = norm(orlTest(:,i)-orlTrain(:,j));
    end
end

cnt=0;

for i=1:N*n
    res(i) = find(Dis(i,:)==min(Dis(i,:)));
    res(i) = floor((res(i)-1)/m)+1;
    if(res(i)==floor((i-1)/n)+1) cnt = cnt+1;
    end
end

cnt/(N*m)

   