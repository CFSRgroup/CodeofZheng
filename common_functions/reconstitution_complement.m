function X=reconstitution_complement(data,N,m,tau)
%该函数用来重构相空间
% m为嵌入空间维数
% tau为时间延迟
% data为输入时间序列
% N为时间序列长度
% X为输出,是m*n维矩阵
M=N-(m-1)*tau;%相空间中点的个数
for j=1:M           %相空间重构
    for i=1:m
        X(i,j)=data((i-1)*tau+j);
    end
end
%数据长度修正
r=size(X,1);
X1=zeros(m,m-1);
X1(1,:)=X(m,end-m+2:end);
X1(end,:)=ones(1,r-1)*X(m,end);
if r>2
for i=2:r
        X1(i,:)=[X(m,end-m+i+1:end) ones(1,i-1)*X(m,end)];
end
end
X=[X X1];
