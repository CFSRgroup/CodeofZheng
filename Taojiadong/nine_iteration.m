err1=sum(D9(errDataH1,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err2=sum(D9(errDataH2,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err3=sum(D9(errDataH3,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����  
err4=sum(D9(errDataH4,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err5=sum(D9(errDataH5,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err6=sum(D9(errDataH6,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err7=sum(D9(errDataH7,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err8=sum(D9(errDataH8,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err9=sum(D9(errDataH9,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
err10=sum(D9(errDataH10,:));%���б��������������Ȩֵ֮�ͼ�Ϊ�����
errAll=[err1;err2;err3;err4;err5;err6;err7;err8;err9;err10]; 
[minErr,minIndex]=min(errAll);  
% ���������e3����G3��ϵ����  
a9=0.5*log((1-minErr)/minErr)  
minErrData=errDataAll(minIndex,:);  
minAccData=accDataAll(minIndex,:);  
D10=D9;  
for i=minAccData'  
    D10(i)=D10(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D10(i)=D10(i)/(2*minErr);  
end  
D10  ;
% ���ຯ��  
f9=a1.*H1+a2*H2+a3*H3+a4*H4+a5*H5+a6*H6+a7*H7+a8*H8+a9*H9;  
HFinal9=sign(f9)%��ʱǿ�������ķ�����  