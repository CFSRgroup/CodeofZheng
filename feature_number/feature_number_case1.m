%NB LR KNN ANN ELM LSSVM SAE
%任务一，8位被试A,B,C,D,E,F,G,H   特征个数按类别逐增
%average_session

clc;
clear;
close all;
warning off;

load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_1
x6=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s7_1
x7=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s8_1
x8=x;
x11=[x1;x2;x3;x4;x5;x6;x7;x8];
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_2
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_2
x6=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s7_2
x7=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s8_2
x8=x;
x22=[x1;x2;x3;x4;x5;x6;x7;x8];
x1=x11;
x2=x22;
y=[y;y;y;y;y;y;y;y];
K=8;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

% %nb
% %%%%%session1
% for j=1
% for i=1:K
%     xtr_1=cell2mat(xtr_all_1(i,:));
%     xte_1=cell2mat(xte_all_1(i,:));
%     ytr_1=cell2mat(ytr_all_1(i,:));
%     yte_1=cell2mat(yte_all_1(i,:));
%     rand('state',0)
%     nb=NaiveBayes.fit(xtr_1(:,1:71),ytr_1); 
%     yte_p=predict(nb,xte_1(:,1:71));
%     cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
%     [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
%     TestingAccuracy(i,j)=Acc(i,j);
%     acc_nb_2_session1=mean(TestingAccuracy);
% end
% end
% 
% %%%session2
% for j=1
% for i=1:K
%     xtr_2=cell2mat(xtr_all_2(i,:));
%     xte_2=cell2mat(xte_all_2(i,:));
%     ytr_2=cell2mat(ytr_all_2(i,:));
%     yte_2=cell2mat(yte_all_2(i,:));
%     rand('state',0)
%     nb=NaiveBayes.fit(xtr_2(:,1:71),ytr_2); 
%     yte_p=predict(nb,xte_2(:,1:71));
%     cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
%     [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
%     TestingAccuracy(i,j)=Acc(i,j);
%     acc_nb_2_session2=mean(TestingAccuracy);
% end
% end
% acc_nb_case1=(acc_nb_2_session1+acc_nb_2_session2)./2;

%elm
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1(:,1:71),ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1(:,1:71),yte_1,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2(:,1:71),ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2(:,1:71),yte_2,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy);
end
end
acc_elm_case1=(acc_elm_2_session1+acc_elm_2_session2)./2;