clc;
clear;
close all;

%K近邻跨被试分类,DEAP数据集
load D:\matlab\matlab7\work\OFS_data_process\article_16\feature_label_DEAP\data

K=32;
%看看正负类样本各有多少
positive=size(find(y_arousal==1));
negative=size(find(y_arousal==0));
[xtr_all, xte_all] = kfcv(x,K,'off');
[ytr_all, yte_all] = kfcv(y_arousal,K,'off');

for j=1
for i=1:K
    xtr=cell2mat(xtr_all(i,:));
    xte=cell2mat(xte_all(i,:));
    ytr=cell2mat(ytr_all(i,:));
    yte=cell2mat(yte_all(i,:));
    
    model=glmfit(xtr,ytr,'binomial','link','logit');
    yte_p=round(glmval(model,xte,'logit'));
    yte_pp(:,i,j)=yte_p;%检查输出是否均为0和1
    
    cte(:,:,i,j)=cfmatrix(yte,yte_p);%计算混淆矩阵
    [sen(i,j),spe(i,j),acc(i,j),pre(i,j),npv(i,j),f1(i,j)] = per_eva(cte(:,:,i,j));
end
end
[para_ind]=find(mean(acc)==max(mean(acc)));%找出最优参数

acc_mean=mean(acc)
f1_mean=mean(f1)

save D:\matlab\matlab7\work\OFS_data_process\article_16\baseline_performance_DEAP\lr_arousal cte sen spe acc pre npv f1 para_ind positive negative