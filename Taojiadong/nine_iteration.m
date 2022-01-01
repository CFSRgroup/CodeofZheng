err1=sum(D9(errDataH1,:));%所有被错分类的样本点的权值之和即为误差率  
err2=sum(D9(errDataH2,:));%所有被错分类的样本点的权值之和即为误差率  
err3=sum(D9(errDataH3,:));%所有被错分类的样本点的权值之和即为误差率  
err4=sum(D9(errDataH4,:));%所有被错分类的样本点的权值之和即为误差率
err5=sum(D9(errDataH5,:));%所有被错分类的样本点的权值之和即为误差率
err6=sum(D9(errDataH6,:));%所有被错分类的样本点的权值之和即为误差率
err7=sum(D9(errDataH7,:));%所有被错分类的样本点的权值之和即为误差率
err8=sum(D9(errDataH8,:));%所有被错分类的样本点的权值之和即为误差率
err9=sum(D9(errDataH9,:));%所有被错分类的样本点的权值之和即为误差率
err10=sum(D9(errDataH10,:));%所有被错分类的样本点的权值之和即为误差率
errAll=[err1;err2;err3;err4;err5;err6;err7;err8;err9;err10]; 
[minErr,minIndex]=min(errAll);  
% 根据误差率e3计算G3的系数：  
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
% 分类函数  
f9=a1.*H1+a2*H2+a3*H3+a4*H4+a5*H5+a6*H6+a7*H7+a8*H8+a9*H9;  
HFinal9=sign(f9)%此时强分类器的分类结果  