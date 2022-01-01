function [r1 r2] = resduv(X,Y)
%���������ڼ������������ʣ�����
N=size(X,1);%�������������������
Rx= similarity_euclid(X);
Rxx=reshape(Rx,N^2,1);
Ry=similarity_euclid(Y);
Ryy=reshape(Ry,N^2,1);
r1=1-corr(Rxx,Ryy)^2;
r2=1-corr(Rxx,Ryy,'type','Spearman')^2;
end

