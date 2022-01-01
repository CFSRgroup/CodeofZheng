err1=sum(D1(errDataH1,:));%所有被错分类的样本点的权值之和即为误差率  
err2=sum(D1(errDataH2,:));%所有被错分类的样本点的权值之和即为误差率  
err3=sum(D1(errDataH3,:));%所有被错分类的样本点的权值之和即为误差率  
err4=sum(D1(errDataH4,:));%所有被错分类的样本点的权值之和即为误差率 
err5=sum(D1(errDataH5,:));%所有被错分类的样本点的权值之和即为误差率
err6=sum(D1(errDataH6,:));%所有被错分类的样本点的权值之和即为误差率
err7=sum(D1(errDataH7,:));%所有被错分类的样本点的权值之和即为误差率
err8=sum(D1(errDataH8,:));%所有被错分类的样本点的权值之和即为误差率
err9=sum(D1(errDataH9,:));%所有被错分类的样本点的权值之和即为误差率
err10=sum(D1(errDataH10,:));%所有被错分类的样本点的权值之和即为误差率
errAll=[err1;err2;err3;err4;err5;err6;err7;err8;err9;err10];  
[minErr,minIndex]=min(errAll);  
%根据误差率e1计算H1的系数：  
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
%分类函数  
f1=a1.*H1;  
HFinal1=sign(f1)%此时强分类器的分类结果  