err1=sum(D8(errDataH1,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err2=sum(D8(errDataH2,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err3=sum(D8(errDataH3,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err4=sum(D8(errDataH4,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err5=sum(D8(errDataH5,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err6=sum(D8(errDataH6,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err7=sum(D8(errDataH7,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err8=sum(D8(errDataH8,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err9=sum(D8(errDataH9,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err10=sum(D8(errDataH10,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
errAll=[err1;err2;err3;err4;err5;err6;err7;err8;err9;err10]; 
[minErr,minIndex]=min(errAll);  
% ���������e3����G3��ϵ����  
a8=0.5*log((1-minErr)/minErr)  
minErrData=errDataAll(minIndex,:);  
minAccData=accDataAll(minIndex,:);  
D9=D8;  
for i=minAccData'  
    D9(i)=D9(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D9(i)=D9(i)/(2*minErr);  
end  
D9  ;
% ���ຯ��  
f8=a1.*H1+a2*H2+a3*H3+a4*H4+a5*H5+a6*H6+a7*H7+a8*H8;  
HFinal8=sign(f8)%��ʱǿ�������ķ�����  