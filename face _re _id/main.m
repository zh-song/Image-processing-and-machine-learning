%orl������
% N = 40; m = 5; n = 5;
% N: ����40������ͼƬ  m:ѵ����������  n:������������
N = 7; m = 7; n = 1; topk = 7;
% ÿһ��Ϊһ������ 

%input
input = im2double(imread('C:\Users\zihao\Desktop\PCA-Face-Recognization-master\dataset\1\00102.bmp'));
if ndims(input)==3
 input = rgb2gray(input);
end
[ox, oy] = size(input);
orlInput = reshape(input, ox*oy, 1);


%��ȡ������Ϣ
for i=1:N
    for j=1:m
        %ԭʼ������Ϣ
        % face{i,j}=im2double(imread(strcat('orl_faces\s',num2str(i),'\s',num2str(i),'-',num2str(j),'.pgm')));
        face{i,j}=im2double(imread(strcat('dataset\',num2str(i),'\00',num2str(i),'0',num2str(j),'.bmp')));
         if ndims(face{i,j})==3
             face{i,j} = rgb2gray(face{i,j});
         end
         [ox, oy] = size(face{i,j});
         orlFace(:,(i-1)*(m)+j) = reshape(face{i,j}, ox*oy, 1);
    end
end

clear i c s cA cH cV cD ox oy dx dy;

%����ѵ�������Լ�����pca��ά
[orlTrain, orlTest] = pcad(orlFace,orlInput);

%����ŷʽ���룬������ڷ�����

for j=1:N*m
    Dis(j) = norm(orlTest-orlTrain(:,j));
end

[B,I] = mink(Dis,topk+1);
for i=1:topk+1
    res(i) = floor((I(i)-1)/m)+1;
    id(i) = mod(I(i),m);
    if id(i) == 0
        id(i) = m;
    end
end
for i=1:topk+1
    img = imread(strcat('dataset\',num2str(res(i)),'\00',num2str(res(i)),'0',num2str(id(i)),'.bmp'));
    subplot(topk+1,1,i),imshow(img)
end
