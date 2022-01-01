%NB LR KNN ANN ELM LSSVM LPP-NB LPP-LR LPP-KNN LPP-ANN LPP-ELM LPP-LSSVM
%H-ELM SAE
%任务一，8位被试A,B,C,D,E,F,G,H   按顺序被试增加  k折交叉验证(8>=k>=2)
%average_session

clc;
clear;
close all;
warning off;

%k=2
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
x11=[x1;x2];
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
x22=[x1;x2];
x1=x11;
x2=x22;
y=[y;y];
K=2;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%h_elm
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session1=mean(TestingAccuracy);
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
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session2=mean(TestingAccuracy);
end
end
acc_h_elm_case1=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case1_h_elm  acc_h_elm_case1

%k=3
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
x11=[x1;x2;x3];
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
x22=[x1;x2;x3];
x1=x11;
x2=x22;
y=[y;y;y];
K=3;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session1=mean(TestingAccuracy);
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
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session2=mean(TestingAccuracy);
end
end
acc_h_elm_case1=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case1_h_elm  acc_h_elm_case1

%k=4
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
x4=x;
x11=[x1;x2;x3;x4];
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
x4=x;
x22=[x1;x2;x3;x4];
x1=x11;
x2=x22;
y=[y;y;y;y];
K=4;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session1=mean(TestingAccuracy);
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
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session2=mean(TestingAccuracy);
end
end
acc_h_elm_case1=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case1_h_elm  acc_h_elm_case1

%k=5
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
x11=[x1;x2;x3;x4;x5];
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
x22=[x1;x2;x3;x4;x5];
x1=x11;
x2=x22;
y=[y;y;y;y;y];
K=5;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session1=mean(TestingAccuracy);
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
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session2=mean(TestingAccuracy);
end
end
acc_h_elm_case1=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case1_h_elm acc_h_elm_case1

%k=6
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
x11=[x1;x2;x3;x4;x5;x6];
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
x22=[x1;x2;x3;x4;x5;x6];
x1=x11;
x2=x22;
y=[y;y;y;y;y;y];
K=6;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session1=mean(TestingAccuracy);
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
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session2=mean(TestingAccuracy);
end
end
acc_h_elm_case1=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case1_h_elm acc_h_elm_case1

%k=7
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
x11=[x1;x2;x3;x4;x5;x6;x7];
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
x22=[x1;x2;x3;x4;x5;x6;x7];
x1=x11;
x2=x22;
y=[y;y;y;y;y;y;y];
K=7;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session1=mean(TestingAccuracy);
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
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session2=mean(TestingAccuracy);
end
end
acc_h_elm_case1=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject7_case1_h_elm acc_h_elm_case1

%k=8
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

for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_1, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session1=mean(TestingAccuracy);
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
sae = saesetup([170 40 40 40 40 40 40 40 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 1;%定义学习率

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 1;%定义学习率

sae.ae{3}.activation_function       = 'sigm';   %第一隐层
sae.ae{3}.learningRate              = 1;%定义学习率

sae.ae{4}.activation_function       = 'sigm';   %第二隐层
sae.ae{4}.learningRate              = 1;%定义学习率

sae.ae{5}.activation_function       = 'sigm';   %第一隐层
sae.ae{5}.learningRate              = 1;%定义学习率

sae.ae{6}.activation_function       = 'sigm';   %第二隐层
sae.ae{6}.learningRate              = 1;%定义学习率

sae.ae{7}.activation_function       = 'sigm';   %第一隐层
sae.ae{7}.learningRate              = 1;%定义学习率

sae.ae{8}.activation_function       = 'sigm';   %第二隐层
sae.ae{8}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 30;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_2, opts);  

[TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
acc_h_elm_2_session2=mean(TestingAccuracy);
end
end
acc_h_elm_case1=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject8_case1_h_elm acc_h_elm_case1
