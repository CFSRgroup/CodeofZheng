%��������OFS���ݴ����������(�����ƶ�ƽ������)
%��д������
clc;
clear;
close all;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_01_eeg'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_01_eog'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_01_ecg'

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_second=Data_process_filter(eeg_data,eog_data,ecg_data,10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eog_feature_01=Data_process_FFT_window(Psych_A_1_second(:,16));

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_02_eeg'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_02_eog'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_02_ecg'

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_second=Data_process_filter(eeg_data,eog_data,ecg_data,10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eog_feature_02=Data_process_FFT_window(Psych_A_1_second(:,16));

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_03_eeg'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_03_eog'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_03_ecg'

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_second=Data_process_filter(eeg_data,eog_data,ecg_data,10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eog_feature_03=Data_process_FFT_window(Psych_A_1_second(:,16));

pp=[eog_feature_01;eog_feature_02;eog_feature_03];

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_04_eeg'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_04_eog'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_04_ecg'

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_second=Data_process_filter(eeg_data,eog_data,ecg_data,10);

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eog_feature_04=Data_process_FFT_window(Psych_A_1_second(:,16));

pp=[eog_feature_01;eog_feature_02;eog_feature_03;eog_feature_04];