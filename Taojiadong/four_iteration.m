err1=sum(D4(errDataH1,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err2=sum(D4(errDataH2,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err3=sum(D4(errDataH3,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err4=sum(D4(errDataH4,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err5=sum(D4(errDataH5,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err6=sum(D4(errDataH6,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err7=sum(D4(errDataH7,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err8=sum(D4(errDataH8,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err9=sum(D4(errDataH9,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err10=sum(D4(errDataH10,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
errAll=[err1;err2;err3;err4;err5;err6;err7;err8;err9;err10]; 
[minErr,minIndex]=min(errAll);  
% ���������e3����G3��ϵ����  
a4=0.5*log((1-minErr)/minErr)  
minErrData=errDataAll(minIndex,:);  
minAccData=accDataAll(minIndex,:);  
D5=D4;  
for i=minAccData'  
    D5(i)=D5(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D5(i)=D5(i)/(2*minErr);  
end  
D5  ;
% ���ຯ��  
f4=a1.*H1+a2.*H2+a3.*H3+a4.*H4;  
HFinal4=sign(f4)%��ʱǿ�������ķ�����  