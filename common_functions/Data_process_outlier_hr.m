function [hr_new]=Data_process_outlier_hr(hr,k)
%本程序用于心率离群点剔除和补零,k表示标准差的倍数,建议取3
%采用均值补零
[p q]=find(hr==0);
for i0=1:length(p)
    hr(p(i0))=mean(hr);
end
%设置sigmak
sigmak=k;
%判断是否存在粗大误差
if max(zscore(hr))>=sigmak | min(zscore(hr))<=-sigmak
    %若误差存在,则进行一维K均值聚类
   [model,data.y]=kmeans(hr,2);
   %找出模型输出索引
   [a b]=find(model==1);
   [c d]=find(model==2);
   [e f]=find(zscore(hr)>=(sigmak) | zscore(hr)<=(-sigmak));
    m=size(a);
    n=size(c);
    hr_new=hr;
    %若第一类样本数目多,则认为其为正常样本
    if m(1)>n(1)
        sum_clear=hr(a(1));
        for i1=1:length(a)-1
            sum_clear=sum_clear+hr(a(i1+1));
        end
        %计算正常数据均值
        ava_clear=sum_clear/length(a);
        %对异常点进行替换
        for i2=1:length(e)
            hr_new(e(i2))=ava_clear;
        end
    else if m(1)<=n(1)
        sum_clear=hr(c(1));
        for i1=1:length(c)-1
            sum_clear=sum_clear+hr(c(i1+1));
        end
        ava_clear=sum_clear/length(c);
        for i2=1:length(e)
            hr_new(e(i2))=ava_clear;
        end
        else
        hr_new=hr;    
        end
    end
else
    hr_new=hr;
end
    
        