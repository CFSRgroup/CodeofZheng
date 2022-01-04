%sdae得出输出，经过Q-statistics，选择新的特征，然后给ELM输入
%任务一挑选4个基学习器，迭代14次；任务二挑选3个基学习器，迭代10次
%对于任务一，首先随机挑选4个基学习器的输出，每两两计算Q值，最后算平均值，重复进行14次，然后排序，绝对值最靠近0的值就选中（选中14次的Q平均值的最接近0值的那一组）
%对于任务二，首先随机挑选3个基学习器的输出，每两两计算Q值，最后算平均值，重复进行10次，然后排序，绝对值最靠近0的值就选中（选中10次的Q平均值的最接近0值的那一组）

clc;
clear;
close all;
warning off;

%case1_session1
for num=1:8
eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session1\Ensemble_member' num2str(num);]);
end

%随机取4个，每两两Q计算（一共算6次），迭代14次
for i=1:14
idx(i,:) = randperm(size(Ensemble_member1,2));
a=Ensemble_member1(:,idx(1:4));

%混淆矩阵
b1=cfmatrix(a(:,1),a(:,2));
b2=cfmatrix(a(:,1),a(:,3));
b3=cfmatrix(a(:,1),a(:,4));
b4=cfmatrix(a(:,2),a(:,3));
b5=cfmatrix(a(:,2),a(:,4));
b6=cfmatrix(a(:,3),a(:,4));

%计算q值
q1=q_statistics(b1);
q2=q_statistics(b2);
q3=q_statistics(b3);
q4=q_statistics(b4);
q5=q_statistics(b5);
q6=q_statistics(b6);

%平均q值
Q(i,:)=(q1+q2+q3+q4+q5+q6)/6;
end

%取平均Q值最小的数，选中该索引为ELM分类器确定训练集
c=min(abs(Q));

rand('state',0)  %resets the generator to its initial state

%elm分类器 
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime, TestingAccuracy, ty] = elm_predict(b1,y_test_DC,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_1=label_index_actual;
    
save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\yte_p\case1_session1\y1 yte_p_1

%case1_session2
for num=1:8
eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member' num2str(num);]);
end

%随机取4个，每两两Q计算（一共算6次），迭代14次
for i=1:14
idx(i,:) = randperm(size(Ensemble_member1,2));
a=Ensemble_member1(:,idx(1:4));

%混淆矩阵
b1=cfmatrix(a(:,1),a(:,2));
b2=cfmatrix(a(:,1),a(:,3));
b3=cfmatrix(a(:,1),a(:,4));
b4=cfmatrix(a(:,2),a(:,3));
b5=cfmatrix(a(:,2),a(:,4));
b6=cfmatrix(a(:,3),a(:,4));

%计算q值
q1=q_statistics(b1);
q2=q_statistics(b2);
q3=q_statistics(b3);
q4=q_statistics(b4);
q5=q_statistics(b5);
q6=q_statistics(b6);

%平均q值
Q(i,:)=(q1+q2+q3+q4+q5+q6)/6;
end

%取平均Q值最小的数，选中该索引为ELM分类器确定训练集
c=min(abs(Q));

rand('state',0)  %resets the generator to its initial state

%elm分类器 
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime, TestingAccuracy, ty] = elm_predict(b1,y_test_DC,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_1=label_index_actual;
    
save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\yte_p\case1_session2\y1 yte_p_1

%case2_session1
for num=1:6
eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session1\Ensemble_member' num2str(num);]);
end

%随机取3个，每两两Q计算（一共算3次），迭代10次
for i=1:10
idx(i,:) = randperm(size(Ensemble_member1,2));
a=Ensemble_member1(:,idx(1:3));

%混淆矩阵
b1=cfmatrix(a(:,1),a(:,2));
b2=cfmatrix(a(:,1),a(:,3));
b3=cfmatrix(a(:,1),a(:,4));

%计算q值
q1=q_statistics(b1);
q2=q_statistics(b2);
q3=q_statistics(b3);

%平均q值
Q(i,:)=(q1+q2+q3)/3;
end

%取平均Q值最小的数，选中该索引为ELM分类器确定训练集
c=min(abs(Q));

rand('state',0)  %resets the generator to its initial state

%elm分类器 
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime, TestingAccuracy, ty] = elm_predict(b1,y_test_DC,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_1=label_index_actual;
    
save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\yte_p\case2_session1\y1 yte_p_1

case2_session2
for num=1:6
eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session2\Ensemble_member' num2str(num);]);
end

%随机取3个，每两两Q计算（一共算3次），迭代10次
for i=1:10
idx(i,:) = randperm(size(Ensemble_member1,2));
a=Ensemble_member1(:,idx(1:3));

%混淆矩阵
b1=cfmatrix(a(:,1),a(:,2));
b2=cfmatrix(a(:,1),a(:,3));
b3=cfmatrix(a(:,1),a(:,4));

%计算q值
q1=q_statistics(b1);
q2=q_statistics(b2);
q3=q_statistics(b3);

%平均q值
Q(i,:)=(q1+q2+q3)/3;
end

%取平均Q值最小的数，选中该索引为ELM分类器确定训练集
c=min(abs(Q));

rand('state',0)  %resets the generator to its initial state

%elm分类器 
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
[TestingTime, TestingAccuracy, ty] = elm_predict(b1,y_test_DC,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_1=label_index_actual;
    
save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\yte_p\case2_session2\y1 yte_p_1
