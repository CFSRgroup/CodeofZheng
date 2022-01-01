%H-ELM跨被试（任务一8位）
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
for j=1:10
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    
   rand('state',0)  %resets the generator to its initial state
%H-ELM分类器
sae = saesetup([170 10*j 10*j 10*j]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

% sae.ae{4}.activation_function       = 'sigm';   %第二隐层
% sae.ae{4}.learningRate              = 1;%定义学习率

% sae.ae{5}.activation_function       = 'sigm';   %第一隐层
% sae.ae{5}.learningRate              = 1;%定义学习率

% sae.ae{6}.activation_function       = 'sigm';   %第二隐层
% sae.ae{6}.learningRate              = 1;%定义学习率

% sae.ae{7}.activation_function       = 'sigm';   %第一隐层
% sae.ae{7}.learningRate              = 1;%定义学习率

% sae.ae{8}.activation_function       = 'sigm';   %第二隐层
% sae.ae{8}.learningRate              = 1;%定义学习率

% sae.ae{9}.activation_function       = 'sigm';   %第一隐层
% sae.ae{9}.learningRate              = 1;%定义学习率

% sae.ae{10}.activation_function       = 'sigm';   %第二隐层
% sae.ae{10}.learningRate              = 1;%定义学习率

% sae.ae{11}.activation_function       = 'sigm';   %第一隐层
% sae.ae{11}.learningRate              = 1;%定义学习率


opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p=label_index_actual;
    
cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;

% [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数

acc_low_mean=mean((acc_low'))';
acc_medium_mean=mean((acc_medium'))';
acc_high_mean=mean((acc_high'))';
Acc_mean=mean((Acc))';  %对角线值除以总值的测试精度
acc_mean=mean((mean_acc'))';
sen_mean=mean((sen'))';
spe_mean=mean((spe'))';
pre_mean=mean((pre'))';
npv_mean=mean((npv'))';
f1_mean=mean((f1'))';
fnr_mean=mean((fnr'))';
fpr_mean=mean((fpr'))';
fdr_mean=mean((fdr'))';
foe_mean=mean((foe'))';
mcc_mean=mean((mcc'))';
bm_mean=mean((bm'))';
mk_mean=mean((mk'))';
ka_mean=mean((ka'))';

acc_TrainingAccuracy=mean((TrainingAccuracy))';
acc_TestingAccuracy=mean((TestingAccuracy))';

% plot_TrainingAccuracy=mean(TrainingAccuracy);
% plot_TestingAccuracy=mean(TestingAccuracy);

% par_ave_mean=mean(mean(TrainingTime'+TestingTime')');
% par_ave_sd=mean(std(TrainingTime'+TestingTime')');

% per_index=[acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean];
accuracy=[acc_TrainingAccuracy acc_TestingAccuracy];
% time=[par_ave_mean par_ave_sd];

% save F:\matlab\trial_procedure\study_1\performance_comparison\h_elm_cross_subject\task1_session1 per_index accuracy time plot_TrainingAccuracy plot_TestingAccuracy
save F:\matlab\trial_procedure\study_1\number_of_abstraction_layers_H-ELM\case1_session1_layer3 Acc_mean
end
end

% subplot(2,1,1);
% plot(plot_TrainingAccuracy,'s-g');
% title('(o) H-ELM: Case 1','FontWeight','bold');
% set(gca,'XTick',0:1:10);
% set(gca,'XTickLabel',{'0','10','20','30','40','50','60','70','80','90','100'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Number of hidden neurons','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy,'o-b');
% hold on;



%%%%session2
for j=1:10
for i=1:K
    
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));

   rand('state',0)  %resets the generator to its initial state
%elm分类器

%H-ELM分类器
sae = saesetup([170 10*j 10*j 10*j]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

% sae.ae{4}.activation_function       = 'sigm';   %第二隐层
% sae.ae{4}.learningRate              = 1;%定义学习率

% sae.ae{5}.activation_function       = 'sigm';   %第一隐层
% sae.ae{5}.learningRate              = 1;%定义学习率

% sae.ae{6}.activation_function       = 'sigm';   %第二隐层
% sae.ae{6}.learningRate              = 1;%定义学习率

% sae.ae{7}.activation_function       = 'sigm';   %第一隐层
% sae.ae{7}.learningRate              = 1;%定义学习率

% sae.ae{8}.activation_function       = 'sigm';   %第二隐层
% sae.ae{8}.learningRate              = 1;%定义学习率

% sae.ae{9}.activation_function       = 'sigm';   %第一隐层
% sae.ae{9}.learningRate              = 1;%定义学习率

% sae.ae{10}.activation_function       = 'sigm';   %第二隐层
% sae.ae{10}.learningRate              = 1;%定义学习率

% sae.ae{11}.activation_function       = 'sigm';   %第一隐层
% sae.ae{11}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p=label_index_actual;
    
cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵

[acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
mean_acc(i,j)=(acc_low(i,j)+acc_medium(i,j)+acc_high(i,j))/3;

% [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数

acc_low_mean=mean((acc_low'))';
acc_medium_mean=mean((acc_medium'))';
acc_high_mean=mean((acc_high'))';
Acc_mean=mean((Acc))';  %对角线值除以总值的测试精度
acc_mean=mean((mean_acc'))';
sen_mean=mean((sen'))';
spe_mean=mean((spe'))';
pre_mean=mean((pre'))';
npv_mean=mean((npv'))';
f1_mean=mean((f1'))';
fnr_mean=mean((fnr'))';
fpr_mean=mean((fpr'))';
fdr_mean=mean((fdr'))';
foe_mean=mean((foe'))';
mcc_mean=mean((mcc'))';
bm_mean=mean((bm'))';
mk_mean=mean((mk'))';
ka_mean=mean((ka'))';

acc_TrainingAccuracy=mean((TrainingAccuracy))';
acc_TestingAccuracy=mean((TestingAccuracy))';

% plot_TrainingAccuracy=mean(TrainingAccuracy);
% plot_TestingAccuracy=mean(TestingAccuracy);
% 
% par_ave_mean=mean(mean(TrainingTime'+TestingTime')');
% par_ave_sd=mean(std(TrainingTime'+TestingTime')');

% per_index=[acc_low_mean,acc_medium_mean,acc_high_mean,Acc_mean,acc_mean,sen_mean,spe_mean,pre_mean,npv_mean,f1_mean,fnr_mean,fpr_mean,fdr_mean,foe_mean,mcc_mean,bm_mean,mk_mean,ka_mean];
accuracy=[acc_TrainingAccuracy acc_TestingAccuracy];
% time=[par_ave_mean par_ave_sd];

% save F:\matlab\trial_procedure\study_1\performance_comparison\h_elm_cross_subject\task1_session2 per_index accuracy time plot_TrainingAccuracy plot_TestingAccuracy
save F:\matlab\trial_procedure\study_1\number_of_abstraction_layers_H-ELM\case1_session2_layer3 Acc_mean

end
end
     
% plot(plot_TrainingAccuracy,'*-r');
% hold on;
% plot(plot_TestingAccuracy,'+-y');
% 
% legend('Training: session 1','Testing: session 1','Training: session 2','Testing: session 2');


%H-ELM跨被试（任务二6位）
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_1
x4_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s5_1
x5_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s6_1
x6_1=x;

x33=[x1_1;x2_1;x3_1;x4_1;x5_1;x6_1];

load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_2
x4_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s5_2
x5_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s6_2
x6_1=x;

x44=[x1_1;x2_1;x3_1;x4_1;x5_1;x6_1];

%任务二两个阶段的数据集
x3=x33;
x4=x44;

y_1=[y;y;y;y;y;y];

K=6;
low=size(find(y_1==1));
medium=size(find(y_1==2));
high=size(find(y_1==3));

[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');

[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%%%%%session1
for j=1:10
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   rand('state',0)  %resets the generator to its initial state
%elm分类器

%H-ELM分类器
sae = saesetup([170 10*j 10*j 10*j]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

% sae.ae{4}.activation_function       = 'sigm';   %第二隐层
% sae.ae{4}.learningRate              = 1;%定义学习率

% sae.ae{5}.activation_function       = 'sigm';   %第一隐层
% sae.ae{5}.learningRate              = 1;%定义学习率

% sae.ae{6}.activation_function       = 'sigm';   %第二隐层
% sae.ae{6}.learningRate              = 1;%定义学习率

% sae.ae{7}.activation_function       = 'sigm';   %第一隐层
% sae.ae{7}.learningRate              = 1;%定义学习率

% sae.ae{8}.activation_function       = 'sigm';   %第二隐层
% sae.ae{8}.learningRate              = 1;%定义学习率

% sae.ae{9}.activation_function       = 'sigm';   %第一隐层
% sae.ae{9}.learningRate              = 1;%定义学习率

% sae.ae{10}.activation_function       = 'sigm';   %第二隐层
% sae.ae{10}.learningRate              = 1;%定义学习率

% sae.ae{11}.activation_function       = 'sigm';   %第一隐层
% sae.ae{11}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data

%elm_kernel分类器
% [ty] = elm_kernel_m(xtr_1, ytr_1, yte_1, xte_1, 1, 2^5, 'lin_kernel', 2^15);

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p_1=label_index_actual_1;
    
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;

% [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数

acc_low_mean_1=mean((acc_low_1'))';
acc_medium_mean_1=mean((acc_medium_1'))';
acc_high_mean_1=mean((acc_high_1'))';
Acc_mean_1=mean((Acc_1))';  %对角线值除以总值的测试精度
acc_mean_1=mean((mean_acc_1'))';
sen_mean_1=mean((sen_1'))';
spe_mean_1=mean((spe_1'))';
pre_mean_1=mean((pre_1'))';
npv_mean_1=mean((npv_1'))';
f1_mean_1=mean((f1_1'))';
fnr_mean_1=mean((fnr_1'))';
fpr_mean_1=mean((fpr_1'))';
fdr_mean_1=mean((fdr_1'))';
foe_mean_1=mean((foe_1'))';
mcc_mean_1=mean((mcc_1'))';
bm_mean_1=mean((bm_1'))';
mk_mean_1=mean((mk_1'))';
ka_mean_1=mean((ka_1'))';

acc_TrainingAccuracy_1=mean((TrainingAccuracy_1))';
acc_TestingAccuracy_1=mean((TestingAccuracy_1))';

% plot_TrainingAccuracy_1=mean(TrainingAccuracy_1);
% plot_TestingAccuracy_1=mean(TestingAccuracy_1);
% 
% par_ave_mean_1=mean(mean(TrainingTime_1'+TestingTime_1')');
% par_ave_sd_1=mean(std(TrainingTime_1'+TestingTime_1')');

% per_index_1=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];
accuracy_1=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
% time_1=[par_ave_mean_1 par_ave_sd_1];

% save F:\matlab\trial_procedure\study_1\performance_comparison\h_elm_cross_subject\task2_session1 per_index_1 accuracy_1 time_1 plot_TrainingAccuracy_1 plot_TestingAccuracy_1
save F:\matlab\trial_procedure\study_1\number_of_abstraction_layers_H-ELM\case2_session1_layer3 Acc_mean_1

end
end

% subplot(2,1,2);
% plot(plot_TrainingAccuracy_1,'s-g');
% title('(p) H-ELM: Case 2','FontWeight','bold');
% set(gca,'XTick',0:1:10);
% set(gca,'XTickLabel',{'0','10','20','30','40','50','60','70','80','90','100'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Number of hidden neurons','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy_1,'o-b');
% hold on;



%%%%%session2
for j=1:10
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));

    rand('state',0)  %resets the generator to its initial state
%elm分类器

%H-ELM分类器
sae = saesetup([170 10*j 10*j 10*j]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

% sae.ae{4}.activation_function       = 'sigm';   %第二隐层
% sae.ae{4}.learningRate              = 1;%定义学习率

% sae.ae{5}.activation_function       = 'sigm';   %第一隐层
% sae.ae{5}.learningRate              = 1;%定义学习率

% sae.ae{6}.activation_function       = 'sigm';   %第二隐层
% sae.ae{6}.learningRate              = 1;%定义学习率

% sae.ae{7}.activation_function       = 'sigm';   %第一隐层
% sae.ae{7}.learningRate              = 1;%定义学习率

% sae.ae{8}.activation_function       = 'sigm';   %第二隐层
% sae.ae{8}.learningRate              = 1;%定义学习率

% sae.ae{9}.activation_function       = 'sigm';   %第一隐层
% sae.ae{9}.learningRate              = 1;%定义学习率

% sae.ae{10}.activation_function       = 'sigm';   %第二隐层
% sae.ae{10}.learningRate              = 1;%定义学习率

% sae.ae{11}.activation_function       = 'sigm';   %第一隐层
% sae.ae{11}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data

%elm_kernel分类器
% [ty] = elm_kernel_m(xtr_1, ytr_1, yte_1, xte_1, 1, 2^5, 'lin_kernel', 2^15);

for zz=1:size(ty_1, 2)
    [~,label_index_actual_1(zz,1)]=max(ty_1(:,zz));
end 
    yte_p_1=label_index_actual_1;
    
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵

[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
mean_acc_1(i,j)=(acc_low_1(i,j)+acc_medium_1(i,j)+acc_high_1(i,j))/3;

% [para_ind]=find(mean_acc==max(mean_acc));%找出最优参数

acc_low_mean_1=mean((acc_low_1'))';
acc_medium_mean_1=mean((acc_medium_1'))';
acc_high_mean_1=mean((acc_high_1'))';
Acc_mean_1=mean((Acc_1))';  %对角线值除以总值的测试精度
acc_mean_1=mean((mean_acc_1'))';
sen_mean_1=mean((sen_1'))';
spe_mean_1=mean((spe_1'))';
pre_mean_1=mean((pre_1'))';
npv_mean_1=mean((npv_1'))';
f1_mean_1=mean((f1_1'))';
fnr_mean_1=mean((fnr_1'))';
fpr_mean_1=mean((fpr_1'))';
fdr_mean_1=mean((fdr_1'))';
foe_mean_1=mean((foe_1'))';
mcc_mean_1=mean((mcc_1'))';
bm_mean_1=mean((bm_1'))';
mk_mean_1=mean((mk_1'))';
ka_mean_1=mean((ka_1'))';

acc_TrainingAccuracy_1=mean((TrainingAccuracy_1))';
acc_TestingAccuracy_1=mean((TestingAccuracy_1))';

% plot_TrainingAccuracy_1=mean(TrainingAccuracy_1);
% plot_TestingAccuracy_1=mean(TestingAccuracy_1);
% 
% par_ave_mean_1=mean(mean(TrainingTime_1'+TestingTime_1')');
% par_ave_sd_1=mean(std(TrainingTime_1'+TestingTime_1')');

% per_index_1=[acc_low_mean_1,acc_medium_mean_1,acc_high_mean_1,Acc_mean_1,acc_mean_1,sen_mean_1,spe_mean_1,pre_mean_1,npv_mean_1,f1_mean_1,fnr_mean_1,fpr_mean_1,fdr_mean_1,foe_mean_1,mcc_mean_1,bm_mean_1,mk_mean_1,ka_mean_1];
accuracy_1=[acc_TrainingAccuracy_1 acc_TestingAccuracy_1];
% time_1=[par_ave_mean_1 par_ave_sd_1];

% save F:\matlab\trial_procedure\study_1\performance_comparison\h_elm_cross_subject\task2_session2 per_index_1 accuracy_1 time_1 plot_TrainingAccuracy_1 plot_TestingAccuracy_1
save F:\matlab\trial_procedure\study_1\number_of_abstraction_layers_H-ELM\case2_session2_layer3 Acc_mean_1

end
end
     
% plot(plot_TrainingAccuracy_1,'*-r');
% hold on;
% plot(plot_TestingAccuracy_1,'+-y');

