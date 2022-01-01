clc;
clear;
close all;

%K���ڿ类�Է���,DEAP���ݼ�
load D:\matlab\matlab7\work\OFS_data_process\article_16\feature_label_DEAP\data

K=32;
%�����������������ж���
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
    
    yte_p=knnclassify(xte,xtr,ytr,5*j);
    cte(:,:,i,j)=cfmatrix(yte,yte_p);%�����������
    [sen(i,j),spe(i,j),acc(i,j),pre(i,j),npv(i,j),f1(i,j)] = per_eva(cte(:,:,i,j));
end
end
[para_ind]=find(mean(acc)==max(mean(acc)));%�ҳ����Ų���

acc_mean=mean(acc)
f1_mean=mean(f1)

save D:\matlab\matlab7\work\OFS_data_process\article_16\baseline_performance_DEAP\knn_arousal cte sen spe acc pre npv f1 para_ind positive negative