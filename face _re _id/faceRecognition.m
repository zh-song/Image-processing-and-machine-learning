%orl������
% N = 40; m = 5; n = 5;
% N: ����40������ͼƬ  m:ѵ����������  n:������������
N = 40; m = 5; n = 1;
% ÿһ��Ϊһ������ 

%��ȡ������Ϣ
for i=1:N
    for j=1:m+n
        %ԭʼ������Ϣ
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

%����ѵ�������Լ�����pca��ά
[orlTrain, orlTest] = dimenReduce(orlFace, m, n);

%����ŷʽ���룬������ڷ�����
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

   