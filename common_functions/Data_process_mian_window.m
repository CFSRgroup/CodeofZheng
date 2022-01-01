%本程序是OFS数据处理的主程序(带有移动平均窗的)
%编写者尹钟
clc;
clear;
close all;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_01_eeg'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_01_eog'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_01_ecg'

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_second=Data_process_filter(eeg_data,eog_data,ecg_data,10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eog_feature_01=Data_process_FFT_window(Psych_A_1_second(:,16));

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_02_eeg'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_02_eog'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_02_ecg'

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_second=Data_process_filter(eeg_data,eog_data,ecg_data,10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eog_feature_02=Data_process_FFT_window(Psych_A_1_second(:,16));

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_03_eeg'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_03_eog'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_03_ecg'

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_second=Data_process_filter(eeg_data,eog_data,ecg_data,10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eog_feature_03=Data_process_FFT_window(Psych_A_1_second(:,16));

pp=[eog_feature_01;eog_feature_02;eog_feature_03];

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_04_eeg'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_04_eog'
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_second_04_ecg'

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_second=Data_process_filter(eeg_data,eog_data,ecg_data,10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eog_feature_04=Data_process_FFT_window(Psych_A_1_second(:,16));

pp=[eog_feature_01;eog_feature_02;eog_feature_03;eog_feature_04];