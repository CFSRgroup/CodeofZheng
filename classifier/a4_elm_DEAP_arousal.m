clc;
clear;
close all;warning off;

%K近邻跨被试分类,DEAP数据集
%load D:\matlab\matlab7\work\OFS_data_process\article_16\feature_label_DEAP\data
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
    
    rand('state',0)  %resets the generator to its initial state
    [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(xtr,ytr+1, 1, j*50, 'sig',2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime, TestingAccuracy, ty] = elm_predict(xte,yte+1,elm_model);  % TY: the actual output of the testing data
    for zz=1:size(ty, 2)
        [~,label_index_actual(zz,1)]=max(ty(:,zz));
    end
    yte_p=label_index_actual-1;
    
    %cte(:,:,i,j)=cfmatrix(yte,yte_p);%计算混淆矩阵
    %[sen(i,j),spe(i,j),acc(i,j),pre(i,j),npv(i,j),f1(i,j)] = per_eva(cte(:,:,i,j));
     %[acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j)]=cla_per(cte(:,:,i,j));
end
end
%[para_ind]=find(mean(acc)==max(mean(acc)));%找出最优参数

%acc_mean=mean(acc)
%f1_mean=mean(f1)

%save D:\matlab\matlab7\work\OFS_data_process\article_16\baseline_performance_DEAP\elm_arousal cte sen spe acc pre npv f1 para_ind positive negative