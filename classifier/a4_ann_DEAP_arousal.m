clc;
clear;
close all;

%K近邻跨被试分类,DEAP数据集
load F:\matlab\trial_procedure\study_1\data

K=32;
%看看正负类样本各有多少
positive=size(find(y_arousal==1));
negative=size(find(y_arousal==0));
[xtr_all, xte_all] = kfcv(x,K,'off');
[ytr_all, yte_all] = kfcv(y_arousal,K,'off');

for j=1:5
for i=1:K
    xtr=cell2mat(xtr_all(i,:));
    xte=cell2mat(xte_all(i,:));
    ytr=cell2mat(ytr_all(i,:));
    yte=cell2mat(yte_all(i,:));
    
    rand('state',0)
    net=patternnet(50*j);
    net=train(net,xtr',ind2vec((ytr+1)'));
    yte_p=net(xte');
    yte_p=vec2ind(yte_p)-1;yte_p=yte_p';
    
    cte(:,:,i,j)=cfmatrix(yte,yte_p);%计算混淆矩阵
    [sen(i,j),spe(i,j),acc(i,j),pre(i,j),npv(i,j),f1(i,j)] = per_eva(cte(:,:,i,j));
end
end
[para_ind]=find(mean(acc)==max(mean(acc)));%找出最优参数

acc_mean=mean(acc)
f1_mean=mean(f1)

%save D:\matlab\matlab7\work\OFS_data_process\article_16\baseline_performance_DEAP\ann_arousal cte sen spe acc pre npv f1 para_ind positive negative