%本程序用于对老实验截取后的心理生理数据进行预处理
%编写者尹钟
clc;
clear;
close all;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_01_all'
%对一个阶段的数据进行滤波
Psych=Data_process_filter(all_data,[],[],10);
%计算EOG=上眼电-下眼电为第12列,将ECG变为第13列,移除第14列
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%可视化前10秒的数据----------------------------------------------------------
figure;%滤波前的数据
for i=1:5
    subplot(6,1,i);
    plot(all_data(1:5000,i));
end
subplot(6,1,1);
title('raw EEG and EOG');
subplot(6,1,6);
plot(all_data(1:5000,13)-all_data(1:5000,12));

figure;%滤波后的数据
for i=1:5
    subplot(6,1,i);
    plot(Psych(1:5000,i));
end
subplot(6,1,1);
title('EEG and EOG after filted');
subplot(6,1,6);
plot(Psych(1:5000,12));
%--------------------------------------------------------------------------

%采用相关法去眼电
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%可视化前2秒的数据----------------------------------------------------------
figure;%滤波后去眼电的5导EEG数据
for i=1:5
    subplot(6,1,i);
    plot(filted_eeg(1:5000,i));
end
subplot(6,1,1);
title('EEG after EOG removal');
subplot(6,1,6);
plot(Psych(1:5000,12));
%--------------------------------------------------------------------------

%将去眼电后的6导EEG替换滤波后的数据
Psych(:,1:6)=filted_eeg;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%删除多余变量
clear all_data

%脑电眼电功率的离群点剔除
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%保存计算后的特征
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%重命名
old_OFS_A_second_1=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_1 old_OFS_A_second_1
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_02_all'
%对一个阶段的数据进行滤波
Psych=Data_process_filter(all_data,[],[],10);
%计算EOG=上眼电-下眼电为第12列,将ECG变为第13列,移除第14列
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%采用相关法去眼电
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych(:,1:6)=filted_eeg;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%删除多余变量
clear all_data

%脑电眼电功率的离群点剔除
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%保存计算后的特征
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%重命名
old_OFS_A_second_2=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_2 old_OFS_A_second_2
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_03_all'
%对一个阶段的数据进行滤波
Psych=Data_process_filter(all_data,[],[],10);
%计算EOG=上眼电-下眼电为第12列,将ECG变为第13列,移除第14列
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%采用相关法去眼电
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych(:,1:6)=filted_eeg;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%删除多余变量
clear all_data

%脑电眼电功率的离群点剔除
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%保存计算后的特征
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%重命名
old_OFS_A_second_3=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_3 old_OFS_A_second_3
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_04_all'
%对一个阶段的数据进行滤波
Psych=Data_process_filter(all_data,[],[],10);
%计算EOG=上眼电-下眼电为第12列,将ECG变为第13列,移除第14列
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%采用相关法去眼电
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych(:,1:6)=filted_eeg;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%删除多余变量
clear all_data

%脑电眼电功率的离群点剔除
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%保存计算后的特征
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%重命名
old_OFS_A_second_4=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_4 old_OFS_A_second_4
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_05_all'
%对一个阶段的数据进行滤波
Psych=Data_process_filter(all_data,[],[],10);
%计算EOG=上眼电-下眼电为第12列,将ECG变为第13列,移除第14列
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%采用相关法去眼电
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych(:,1:6)=filted_eeg;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%删除多余变量
clear all_data

%脑电眼电功率的离群点剔除
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%保存计算后的特征
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%重命名
old_OFS_A_second_5=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_5 old_OFS_A_second_5
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_06_all'
%对一个阶段的数据进行滤波
Psych=Data_process_filter(all_data,[],[],10);
%计算EOG=上眼电-下眼电为第12列,将ECG变为第13列,移除第14列
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%采用相关法去眼电
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych(:,1:6)=filted_eeg;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%删除多余变量
clear all_data

%脑电眼电功率的离群点剔除
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%保存计算后的特征
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%重命名
old_OFS_A_second_6=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_6 old_OFS_A_second_6
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_07_all'
%对一个阶段的数据进行滤波
Psych=Data_process_filter(all_data,[],[],10);
%计算EOG=上眼电-下眼电为第12列,将ECG变为第13列,移除第14列
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%采用相关法去眼电
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych(:,1:6)=filted_eeg;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%删除多余变量
clear all_data

%脑电眼电功率的离群点剔除
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%保存计算后的特征
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%重命名
old_OFS_A_second_7=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_7 old_OFS_A_second_7
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_08_all'
%对一个阶段的数据进行滤波
Psych=Data_process_filter(all_data,[],[],10);
%计算EOG=上眼电-下眼电为第12列,将ECG变为第13列,移除第14列
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%采用相关法去眼电
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych(:,1:6)=filted_eeg;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%删除多余变量
clear all_data

%脑电眼电功率的离群点剔除
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%保存计算后的特征
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%重命名
old_OFS_A_second_8=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_8 old_OFS_A_second_8
%--------------------------------------------------------------------------