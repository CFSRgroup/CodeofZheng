%���������ڶ���ʵ���ȡ��������������ݽ���Ԥ����
%��д������
clc;
clear;
close all;

%%
%��������
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_01_all'

%��ȡ��׼���ȵ�����
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(eeg,[],[],10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_feature=Data_process_FFT(Psych);

%�Ե��۵繦�ʵ���Ⱥ���޳�
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%������
p1=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_1 p1
%--------------------------------------------------------------------------
%%
%��������
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_02_all'

%��ȡ��׼���ȵ�����
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(eeg,[],[],10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_feature=Data_process_FFT(Psych);

%�Ե��۵繦�ʵ���Ⱥ���޳�
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%������
p2=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_2 p2

PD_D_second_1=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_1 PD_D_second_1
%--------------------------------------------------------------------------
%%
%��������
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_03_all'

%��ȡ��׼���ȵ�����
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(eeg,[],[],10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_feature=Data_process_FFT(Psych);

%�Ե��۵繦�ʵ���Ⱥ���޳�
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%������
p3=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_3 p3

PD_D_second_2=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_2 PD_D_second_2
%--------------------------------------------------------------------------
%%
%��������
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_04_all'

%��ȡ��׼���ȵ�����
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(eeg,[],[],10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_feature=Data_process_FFT(Psych);

%�Ե��۵繦�ʵ���Ⱥ���޳�
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%������
p4=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_4 p4

PD_D_second_3=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_3 PD_D_second_3
%--------------------------------------------------------------------------
%%
%��������
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_05_all'

%��ȡ��׼���ȵ�����
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(eeg,[],[],10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_feature=Data_process_FFT(Psych);

%�Ե��۵繦�ʵ���Ⱥ���޳�
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%������
p5=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_5 p5

PD_D_second_4=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_4 PD_D_second_4
%--------------------------------------------------------------------------
%%
%��������
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_06_all'

%��ȡ��׼���ȵ�����
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(eeg,[],[],10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_feature=Data_process_FFT(Psych);

%�Ե��۵繦�ʵ���Ⱥ���޳�
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%������
p6=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_6 p6

PD_D_second_5=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_5 PD_D_second_5
%--------------------------------------------------------------------------
%%
%��������
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_07_all'

%��ȡ��׼���ȵ�����
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(eeg,[],[],10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_feature=Data_process_FFT(Psych);

%�Ե��۵繦�ʵ���Ⱥ���޳�
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%������
p7=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_7 p7

PD_D_second_6=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_6 PD_D_second_6
%--------------------------------------------------------------------------
%%
%��������
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_08_all'

%��ȡ��׼���ȵ�����
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(eeg,[],[],10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_feature=Data_process_FFT(Psych);

%�Ե��۵繦�ʵ���Ⱥ���޳�
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%������
p8=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_8 p8
%--------------------------------------------------------------------------