function X=reconstitution_complement(data,N,m,tau)
%�ú��������ع���ռ�
% mΪǶ��ռ�ά��
% tauΪʱ���ӳ�
% dataΪ����ʱ������
% NΪʱ�����г���
% XΪ���,��m*nά����
M=N-(m-1)*tau;%��ռ��е�ĸ���
for j=1:M           %��ռ��ع�
    for i=1:m
        X(i,j)=data((i-1)*tau+j);
    end
end
%���ݳ�������
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
