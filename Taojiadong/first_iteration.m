err1=sum(D1(errDataH1,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err2=sum(D1(errDataH2,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err3=sum(D1(errDataH3,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err4=sum(D1(errDataH4,:));%���б��������������Ȩֵ֮�ͼ�Ϊ����� 
err5=sum(D1(errDataH5,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err6=sum(D1(errDataH6,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err7=sum(D1(errDataH7,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err8=sum(D1(errDataH8,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err9=sum(D1(errDataH9,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err10=sum(D1(errDataH10,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
errAll=[err1;err2;err3;err4;err5;err6;err7;err8;err9;err10];  
[minErr,minIndex]=min(errAll);  
%���������e1����H1��ϵ����  
a1=0.5*log((1-minErr)/minErr)  
minErrData=errDataAll(minIndex,:);  
minAccData=accDataAll(minIndex,:);  
D2=D1;  
for i=minAccData'  
    D2(i)=D2(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D2(i)=D2(i)/(2*minErr);  
end  
D2  ;
%���ຯ��  
f1=a1.*H1;  
HFinal1=sign(f1)%��ʱǿ�������ķ�����  