%%%%%%%
%ELM集成跨被试（任务一8位）
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

%任务一两个阶段的数据集
x1=x11;
x2=x22;

y=[y;y;y;y;y;y;y;y];

K=8;
low=size(find(y==1));
medium=size(find(y==2));
high=size(find(y==3));

[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');

[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    
   rand('state',0)  %resets the generator to its initial state
%elm分类器
%7位被试建模
[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_1(1:2700,:),ytr_1(1:2700,:), 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TrainingTime_2(i,j),TrainingAccuracy_2(i,j),elm_model_2,label_index_expected_2] = elm_train_m(xtr_1(2701:5400,:),ytr_1(2701:5400,:), 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TrainingTime_3(i,j),TrainingAccuracy_3(i,j),elm_model_3,label_index_expected_3] = elm_train_m(xtr_1(5401:8100,:),ytr_1(5401:8100,:), 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TrainingTime_4(i,j),TrainingAccuracy_4(i,j),elm_model_4,label_index_expected_4] = elm_train_m(xtr_1(8101:10800,:),ytr_1(8101:10800,:), 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TrainingTime_5(i,j),TrainingAccuracy_5(i,j),elm_model_5,label_index_expected_5] = elm_train_m(xtr_1(10801:13500,:),ytr_1(10801:13500,:), 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TrainingTime_6(i,j),TrainingAccuracy_6(i,j),elm_model_6,label_index_expected_6] = elm_train_m(xtr_1(13501:16200,:),ytr_1(13501:16200,:), 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TrainingTime_7(i,j),TrainingAccuracy_7(i,j),elm_model_7,label_index_expected_7] = elm_train_m(xtr_1(16201:18900,:),ytr_1(16201:18900,:), 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数

%1位被试测试，7组结果，平均值法
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_1,yte_1,elm_model_1);  % TY: the actual output of the testing data
[TestingTime_2(i,j), TestingAccuracy_2(i,j), ty_2] = elm_predict(xte_1,yte_1,elm_model_2);  % TY: the actual output of the testing data
[TestingTime_3(i,j), TestingAccuracy_3(i,j), ty_3] = elm_predict(xte_1,yte_1,elm_model_3);  % TY: the actual output of the testing data
[TestingTime_4(i,j), TestingAccuracy_4(i,j), ty_4] = elm_predict(xte_1,yte_1,elm_model_4);  % TY: the actual output of the testing data
[TestingTime_5(i,j), TestingAccuracy_5(i,j), ty_5] = elm_predict(xte_1,yte_1,elm_model_5);  % TY: the actual output of the testing data
[TestingTime_6(i,j), TestingAccuracy_6(i,j), ty_6] = elm_predict(xte_1,yte_1,elm_model_6);  % TY: the actual output of the testing data
[TestingTime_7(i,j), TestingAccuracy_7(i,j), ty_7] = elm_predict(xte_1,yte_1,elm_model_7);  % TY: the actual output of the testing data

for zz_1=1:size(ty_1, 2)
    [~,label_index_actual_1(zz_1,1)]=max(ty_1(:,zz_1));
end 
    yte_p_1=label_index_actual_1;

for zz_2=1:size(ty_2, 2)
    [~,label_index_actual_2(zz_2,1)]=max(ty_2(:,zz_2));
end 
    yte_p_2=label_index_actual_2;

for zz_3=1:size(ty_3, 2)
    [~,label_index_actual_3(zz_3,1)]=max(ty_3(:,zz_3));
end 
    yte_p_3=label_index_actual_3;    

for zz_4=1:size(ty_4, 2)
    [~,label_index_actual_4(zz_4,1)]=max(ty_4(:,zz_4));
end 
    yte_p_4=label_index_actual_4;

for zz_5=1:size(ty_5, 2)
    [~,label_index_actual_5(zz_5,1)]=max(ty_5(:,zz_5));
end 
    yte_p_5=label_index_actual_5;

for zz_6=1:size(ty_6, 2)
    [~,label_index_actual_6(zz_6,1)]=max(ty_6(:,zz_6));
end 
    yte_p_6=label_index_actual_6;

for zz_7=1:size(ty_7, 2)
    [~,label_index_actual_7(zz_7,1)]=max(ty_7(:,zz_7));
end 
    yte_p_7=label_index_actual_7;

yte_pp=[yte_p_1 yte_p_2 yte_p_3 yte_p_4 yte_p_5 yte_p_6 yte_p_7];

% yte_p=round(mean(yte_pp'))';
yte_p=mode(yte_pp')';
    
cte(:,:,i,j)=cfmatrix(yte_1,yte_pp(:,7));%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;





% acc_TrainingAccuracy=mean(TrainingAccuracy); %8位被试平均训练精度和测试精度
% acc_TestingAccuracy=mean(TestingAccuracy);
% 
% par_ave_mean=mean(TrainingTime+TestingTime); %8位被试平均时间和时间偏差
% par_ave_sd=std(TrainingTime+TestingTime);
% 
% accuracy_par_ave_mean=acc_TestingAccuracy;
% accuracy_par_ave_sd=std(TestingAccuracy);
% 
% subject_average_accuracy1=[accuracy_par_ave_mean accuracy_par_ave_sd]; %重复50次实验
% per_index_single1=[acc_low,acc_medium,acc_high,Acc,acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka];
% per_index_ave1=mean([acc_low,acc_medium,acc_high,Acc,acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka])';
% accuracy1=[acc_TrainingAccuracy acc_TestingAccuracy];
% time1=[par_ave_mean par_ave_sd];

end
end