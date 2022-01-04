%sdae�ó����������Q-statistics��ѡ���µ�������Ȼ���ELM����
%����һ��ѡ4����ѧϰ��������14�Σ��������ѡ3����ѧϰ��������10��
%��������һ�����������ѡ4����ѧϰ���������ÿ��������Qֵ�������ƽ��ֵ���ظ�����14�Σ�Ȼ�����򣬾���ֵ���0��ֵ��ѡ�У�ѡ��14�ε�Qƽ��ֵ����ӽ�0ֵ����һ�飩
%��������������������ѡ3����ѧϰ���������ÿ��������Qֵ�������ƽ��ֵ���ظ�����10�Σ�Ȼ�����򣬾���ֵ���0��ֵ��ѡ�У�ѡ��10�ε�Qƽ��ֵ����ӽ�0ֵ����һ�飩

clc;
clear;
close all;
warning off;

%case1_session1
for num=1:8
eval(['load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session1\Ensemble_member' num2str(num);]);
end

%���ȡ4����ÿ����Q���㣨һ����6�Σ�������14��
for i=1:14
idx(i,:) = randperm(size(Ensemble_member1,2));
a=Ensemble_member1(:,idx(1:4));

%��������
b1=cfmatrix(a(:,1),a(:,2));
b2=cfmatrix(a(:,1),a(:,3));
b3=cfmatrix(a(:,1),a(:,4));
b4=cfmatrix(a(:,2),a(:,3));
b5=cfmatrix(a(:,2),a(:,4));
b6=cfmatrix(a(:,3),a(:,4));

%����qֵ
q1=q_statistics(b1);
q2=q_statistics(b2);
q3=q_statistics(b3);
q4=q_statistics(b4);
q5=q_statistics(b5);
q6=q_statistics(b6);

%ƽ��qֵ
Q(i,:)=(q1+q2+q3+q4+q5+q6)/6;
end

%ȡƽ��Qֵ��С������ѡ�и�����ΪELM������ȷ��ѵ����
c=min(abs(Q));

rand('state',0)  %resets the generator to its initial state

%elm������ 
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 390, 'sig', 2^0);%ELMֻ��1��2���ǩ������0��1   elm_train_m()���һ���������Ǽ��������
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

%���ȡ4����ÿ����Q���㣨һ����6�Σ�������14��
for i=1:14
idx(i,:) = randperm(size(Ensemble_member1,2));
a=Ensemble_member1(:,idx(1:4));

%��������
b1=cfmatrix(a(:,1),a(:,2));
b2=cfmatrix(a(:,1),a(:,3));
b3=cfmatrix(a(:,1),a(:,4));
b4=cfmatrix(a(:,2),a(:,3));
b5=cfmatrix(a(:,2),a(:,4));
b6=cfmatrix(a(:,3),a(:,4));

%����qֵ
q1=q_statistics(b1);
q2=q_statistics(b2);
q3=q_statistics(b3);
q4=q_statistics(b4);
q5=q_statistics(b5);
q6=q_statistics(b6);

%ƽ��qֵ
Q(i,:)=(q1+q2+q3+q4+q5+q6)/6;
end

%ȡƽ��Qֵ��С������ѡ�и�����ΪELM������ȷ��ѵ����
c=min(abs(Q));

rand('state',0)  %resets the generator to its initial state

%elm������ 
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 390, 'sig', 2^0);%ELMֻ��1��2���ǩ������0��1   elm_train_m()���һ���������Ǽ��������
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

%���ȡ3����ÿ����Q���㣨һ����3�Σ�������10��
for i=1:10
idx(i,:) = randperm(size(Ensemble_member1,2));
a=Ensemble_member1(:,idx(1:3));

%��������
b1=cfmatrix(a(:,1),a(:,2));
b2=cfmatrix(a(:,1),a(:,3));
b3=cfmatrix(a(:,1),a(:,4));

%����qֵ
q1=q_statistics(b1);
q2=q_statistics(b2);
q3=q_statistics(b3);

%ƽ��qֵ
Q(i,:)=(q1+q2+q3)/3;
end

%ȡƽ��Qֵ��С������ѡ�и�����ΪELM������ȷ��ѵ����
c=min(abs(Q));

rand('state',0)  %resets the generator to its initial state

%elm������ 
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 60, 'sig', 2^0);%ELMֻ��1��2���ǩ������0��1   elm_train_m()���һ���������Ǽ��������
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

%���ȡ3����ÿ����Q���㣨һ����3�Σ�������10��
for i=1:10
idx(i,:) = randperm(size(Ensemble_member1,2));
a=Ensemble_member1(:,idx(1:3));

%��������
b1=cfmatrix(a(:,1),a(:,2));
b2=cfmatrix(a(:,1),a(:,3));
b3=cfmatrix(a(:,1),a(:,4));

%����qֵ
q1=q_statistics(b1);
q2=q_statistics(b2);
q3=q_statistics(b3);

%ƽ��qֵ
Q(i,:)=(q1+q2+q3)/3;
end

%ȡƽ��Qֵ��С������ѡ�и�����ΪELM������ȷ��ѵ����
c=min(abs(Q));

rand('state',0)  %resets the generator to its initial state

%elm������ 
[TrainingTime,TrainingAccuracy,elm_model,label_index_expected] = elm_train_m(Ensemble_member1,y_test_BC1, 1, 60, 'sig', 2^0);%ELMֻ��1��2���ǩ������0��1   elm_train_m()���һ���������Ǽ��������
[TestingTime, TestingAccuracy, ty] = elm_predict(b1,y_test_DC,elm_model);  % TY: the actual output of the testing data

for zz=1:size(ty, 2)
    [~,label_index_actual(zz,1)]=max(ty(:,zz));
end 
    yte_p_1=label_index_actual;
    
save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\yte_p\case2_session2\y1 yte_p_1
