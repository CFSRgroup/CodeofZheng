%集成输入给ELM训练，交叉验证测试集给ELM测试，最终得8组解
clc;
clear;
close all;
warning off;

% %case1_session1
% for num=1:8
% eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session1\Ensemble_member' num2str(num);]);
% end

% %case1_session2
% for num=1:8
% eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member' num2str(num);]);
% end
% 
% %case2_session1
% for num=1:6
% eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session1\Ensemble_member' num2str(num);]);
% end
% 
%case2_session2
for num=1:6
eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session2\Ensemble_member' num2str(num);]);
end

% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session1\BC_test
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session1\DC_test

% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session2\BC_test_2
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session2\DC_test_2
% % 
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session1\BC_test_3
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session1\DC_test_3

load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session2\BC_test_4
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session2\DC_test_4

%随机取Ensemble_member 7列中4列
n=3;
idx=randperm(5);
idx=idx(1:n);
Ensemble_member1=Ensemble_member1(:,idx);
Ensemble_member2=Ensemble_member2(:,idx);
Ensemble_member3=Ensemble_member3(:,idx);
Ensemble_member4=Ensemble_member4(:,idx);
Ensemble_member5=Ensemble_member5(:,idx);
Ensemble_member6=Ensemble_member6(:,idx);
% Ensemble_member7=Ensemble_member7(:,idx);
% Ensemble_member8=Ensemble_member8(:,idx);

b1=b1(:,idx);
b2=b2(:,idx);
b3=b3(:,idx);
b4=b4(:,idx);
b5=b5(:,idx);
b6=b6(:,idx);
% b7=b7(:,idx);
% b8=b8(:,idx);

% rand('state',0)  %resets the generator to its initial state

% %elm分类器   A
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b1,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_A=TrainingTime+TestingTime+Time1;
% 
% per_index_A=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case1_session2\A per_index_A time_A

% %elm分类器  B
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member2,y_test_BC2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b2,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_B=TrainingTime+TestingTime+Time2;
% 
% per_index_B=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case1_session2\B per_index_B time_B


% %elm分类器  C
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member3,y_test_BC3, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b3,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_C=TrainingTime+TestingTime+Time3;
% 
% per_index_C=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case1_session2\C per_index_C time_C

% 
% %elm分类器  D
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member4,y_test_BC4, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b4,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_D=TrainingTime+TestingTime+Time4;
% 
% per_index_D=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case1_session2\D per_index_D time_D
% 
% 
% %elm分类器  E
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member5,y_test_BC5, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b5,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_E=TrainingTime+TestingTime+Time5;
% 
% per_index_E=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case1_session2\E per_index_E time_E
% 
% 
% %elm分类器   F
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member6,y_test_BC6, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b6,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_F=TrainingTime+TestingTime+Time6;
% 
% per_index_F=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case1_session2\F per_index_F time_F
% 
% 
% % elm分类器  G
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member7,y_test_BC7, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b7,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_G=TrainingTime+TestingTime+Time7;
% 
% per_index_G=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case1_session2\G per_index_G time_G


% %elm分类器   H
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member8,y_test_BC8, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b8,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_H=TrainingTime+TestingTime+Time8;
% 
% per_index_H=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case1_session2\H per_index_H time_H




%case2_session1
% %elm分类器   I
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b1,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_I=TrainingTime+TestingTime+Time1;
% 
% per_index_I=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case2_session2\I per_index_I time_I

% %elm分类器  J
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member2,y_test_BC2, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b2,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_J=TrainingTime+TestingTime+Time2;
% per_index_J=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case2_session2\J per_index_J time_J


% %elm分类器  K
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member3,y_test_BC3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b3,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_K=TrainingTime+TestingTime+Time3;
% 
% per_index_K=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case2_session2\K per_index_K time_K


% %elm分类器  L
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member4,y_test_BC4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b4,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_L=TrainingTime+TestingTime+Time4;
% 
% per_index_L=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case2_session2\L per_index_L time_L

% 
% %elm分类器  M
% [TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member5,y_test_BC5, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
% [TestingTime, TestingAccuracy, ty] = elm_predict(b5,y_test_DC,elm_model);  % TY: the actual output of the testing data
% 
% for zz=1:size(ty, 2)
%     [~,label_index_actual(zz,1)]=max(ty(:,zz));
% end 
%     yte_p=label_index_actual;
%     
% cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% time_M=TrainingTime+TestingTime+Time5;
% 
% per_index_M=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case2_session2\M per_index_M time_M
% 
% 
%elm分类器   N
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member6,y_test_BC6, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime, TestingAccuracy, ty] = elm_predict(b6,y_test_DC,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p=label_index_actual;
    
cte=cfmatrix(y_test_DC,yte_p);%计算混淆矩阵
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(cte); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
time_N=TrainingTime+TestingTime+Time6;

per_index_N=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\per_index\case2_session2\N per_index_N time_N








