%本程序用于对老实验截取后的心理生理数据进行预处理
%编写者尹钟
clc;
clear;
close all;

%%
%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_01_all'

%截取标准长度的数据
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%对一个阶段的数据进行滤波
Psych=Data_process_filter(eeg,[],[],10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_feature=Data_process_FFT(Psych);

%脑电眼电功率的离群点剔除
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%重命名
p1=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_1 p1
%--------------------------------------------------------------------------
%%
%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_02_all'

%截取标准长度的数据
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%对一个阶段的数据进行滤波
Psych=Data_process_filter(eeg,[],[],10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_feature=Data_process_FFT(Psych);

%脑电眼电功率的离群点剔除
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%重命名
p2=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_2 p2

PD_D_second_1=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_1 PD_D_second_1
%--------------------------------------------------------------------------
%%
%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_03_all'

%截取标准长度的数据
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%对一个阶段的数据进行滤波
Psych=Data_process_filter(eeg,[],[],10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_feature=Data_process_FFT(Psych);

%脑电眼电功率的离群点剔除
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%重命名
p3=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_3 p3

PD_D_second_2=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_2 PD_D_second_2
%--------------------------------------------------------------------------
%%
%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_04_all'

%截取标准长度的数据
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%对一个阶段的数据进行滤波
Psych=Data_process_filter(eeg,[],[],10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_feature=Data_process_FFT(Psych);

%脑电眼电功率的离群点剔除
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%重命名
p4=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_4 p4

PD_D_second_3=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_3 PD_D_second_3
%--------------------------------------------------------------------------
%%
%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_05_all'

%截取标准长度的数据
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%对一个阶段的数据进行滤波
Psych=Data_process_filter(eeg,[],[],10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_feature=Data_process_FFT(Psych);

%脑电眼电功率的离群点剔除
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%重命名
p5=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_5 p5

PD_D_second_4=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_4 PD_D_second_4
%--------------------------------------------------------------------------
%%
%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_06_all'

%截取标准长度的数据
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%对一个阶段的数据进行滤波
Psych=Data_process_filter(eeg,[],[],10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_feature=Data_process_FFT(Psych);

%脑电眼电功率的离群点剔除
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%重命名
p6=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_6 p6

PD_D_second_5=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_5 PD_D_second_5
%--------------------------------------------------------------------------
%%
%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_07_all'

%截取标准长度的数据
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%对一个阶段的数据进行滤波
Psych=Data_process_filter(eeg,[],[],10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_feature=Data_process_FFT(Psych);

%脑电眼电功率的离群点剔除
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%重命名
p7=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_7 p7

PD_D_second_6=xperf;

save D:\matlab\matlab7\work\OFS_data_process\OFS_performance_new_old\PD_D_second_6 PD_D_second_6
%--------------------------------------------------------------------------
%%
%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_D\OFS_chenchao_second_08_all'

%截取标准长度的数据
eeg=eeg(1:end-2000,1:11);
eeg=double(eeg);
%对一个阶段的数据进行滤波
Psych=Data_process_filter(eeg,[],[],10);

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_feature=Data_process_FFT(Psych);

%脑电眼电功率的离群点剔除
eeg_feature_new=zeros(size(eeg_feature,1),size(eeg_feature,2));
for i=1:440
    eeg_feature_new(:,i)=Data_process_outlier_hr(eeg_feature(:,i),3);
end

%重命名
p8=[eeg_feature_new HRtime];

save D:\matlab\matlab7\work\OFS_data_process\new_old_OFS_feature\new_old_OFS_D_second_8 p8
%--------------------------------------------------------------------------