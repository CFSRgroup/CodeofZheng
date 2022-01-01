clc;
clear;
close all;
warning off;

load F:\matlab\trial_procedure\study_1\features\ex1\s1_1

K=10;
%看看low,medium,high类样本各有多少
low=size(find(y==1));
medium=size(find(y==2));
high=size(find(y==3));
%high=size(find(y==2))+size(find(y==3));
[xtr_all, xte_all] = kfcv(x,K,'off');
[ytr_all, yte_all] = kfcv(y,K,'off');

for j=1:5
for i=1:K
    xtr=cell2mat(xtr_all(i,:));
    xte=cell2mat(xte_all(i,:));
    ytr=cell2mat(ytr_all(i,:));
    yte=cell2mat(yte_all(i,:));
    
    rand('state',0)  %resets the generator to its initial state
    [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(xtr,ytr, 1, j*50, 'sig',2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime, TestingAccuracy, ty] = elm_predict(xte,yte,elm_model);  % TY: the actual output of the testing data
    
    for zz=1:size(ty, 2)
       [~,label_index_actual(zz,1)]=max(ty(:,zz));
    end 
    yte_p=label_index_actual;

    %cte(:,:,i,j)=cfmatrix(yte,yte_p);%计算混淆矩阵
    %[sen(i,j),spe(i,j),acc(i,j),pre(i,j),npv(i,j),f1(i,j)] = per_eva(cte(:,:,i,j));
    %[acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j)]=cla_per(cte(:,:,i,j));
end
end