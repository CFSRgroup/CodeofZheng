function [r1 r2] = resduv(X,Y)
%本函数用于计算两个矩阵的剩余误差
N=size(X,1);%计算输入矩阵样本个数
Rx= similarity_euclid(X);
Rxx=reshape(Rx,N^2,1);
Ry=similarity_euclid(Y);
Ryy=reshape(Ry,N^2,1);
r1=1-corr(Rxx,Ryy)^2;
r2=1-corr(Rxx,Ryy,'type','Spearman')^2;
end

