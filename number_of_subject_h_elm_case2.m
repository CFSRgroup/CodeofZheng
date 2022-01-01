%h_elm
%任务二，6位被试I,J,K,L,M,N   按顺序被试增加  k折交叉验证(6>=k>=2)
%average_session

clc;
clear;
close all;
warning off;

%k=2
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
x33=[x1_1;x2_1];
load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
x44=[x1_1;x2_1];
x3=x33;
x4=x44;
y_1=[y;y];
K=2;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%h_elm
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
    
    sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data
 acc_h_elm_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
    
sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
    acc_h_elm_2_session2=mean(TestingAccuracy_1);
end
end
acc_h_elm_case2=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case2_h_elm  acc_h_elm_case2


%k=3
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
x3_1=x;
x33=[x1_1;x2_1;x3_1];
load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
x3_1=x;
x44=[x1_1;x2_1;x3_1];
x3=x33;
x4=x44;
y_1=[y;y;y];
K=3;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%h_elm
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
    
    sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data
 acc_h_elm_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
    
sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
    acc_h_elm_2_session2=mean(TestingAccuracy_1);
end
end
acc_h_elm_case2=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case2_h_elm  acc_h_elm_case2


%k=4
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_1
x4_1=x;
x33=[x1_1;x2_1;x3_1;x4_1];
load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_2
x4_1=x;
x44=[x1_1;x2_1;x3_1;x4_1];
x3=x33;
x4=x44;
y_1=[y;y;y;y];
K=4;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%h_elm
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
    
    sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data
 acc_h_elm_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
    
sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
    acc_h_elm_2_session2=mean(TestingAccuracy_1);
end
end
acc_h_elm_case2=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case2_h_elm  acc_h_elm_case2

%k=5
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
x33=[x1_1;x2_1;x3_1;x4_1;x5_1];
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
x44=[x1_1;x2_1;x3_1;x4_1;x5_1];
x3=x33;
x4=x44;
y_1=[y;y;y;y;y];
K=5;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%h_elm
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
    
    sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data
 acc_h_elm_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
    
sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
    acc_h_elm_2_session2=mean(TestingAccuracy_1);
end
end
acc_h_elm_case2=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case2_h_elm  acc_h_elm_case2

%k=6
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
x3=x33;
x4=x44;
y_1=[y;y;y;y;y;y];
K=6;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%h_elm
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
    
    sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3,ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3,yte_3,elm_model_1);  % TY: the actual output of the testing data
 acc_h_elm_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
    
sae = saesetup([170 10 10 10 10 10 10 10 10 10]);
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

sae.ae{9}.activation_function       = 'sigm';   %第一隐层
sae.ae{9}.learningRate              = 1;%定义学习率

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);  

[TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4,ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4,yte_4,elm_model_1);  % TY: the actual output of the testing data
    acc_h_elm_2_session2=mean(TestingAccuracy_1);
end
end
acc_h_elm_case2=(acc_h_elm_2_session1+acc_h_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case2_h_elm  acc_h_elm_case2

