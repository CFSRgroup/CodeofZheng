err1=sum(D5(errDataH1,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err2=sum(D5(errDataH2,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err3=sum(D5(errDataH3,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err4=sum(D5(errDataH4,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err5=sum(D5(errDataH5,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err6=sum(D5(errDataH6,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err7=sum(D5(errDataH7,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
 err8=sum(D5(errDataH8,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err9=sum(D5(errDataH9,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err10=sum(D5(errDataH10,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
errAll=[err1;err2;err3;err4;err5;err6;err7;err8;err9;err10]; 
[minErr,minIndex]=min(errAll);  
% ���������e3����G3��ϵ����  
a5=0.5*log((1-minErr)/minErr)  
minErrData=errDataAll(minIndex,:);  
minAccData=accDataAll(minIndex,:);  
D6=D5;  
for i=minAccData'  
    D6(i)=D6(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D6(i)=D6(i)/(2*minErr);  
end  
D6  ;
% ���ຯ��  
f5=a1.*H1+a2*H2+a3*H3+a4*H4+a5*H5;  
HFinal5=sign(f5)%��ʱǿ�������ķ�����  