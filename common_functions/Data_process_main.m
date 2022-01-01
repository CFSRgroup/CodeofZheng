%本程序是OFS数据处理的主程序
%编写者尹钟
clc;
clear;
close all;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_01_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_01_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_01_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%可视化前10秒的数据----------------------------------------------------------
figure;%滤波前的数据
for i=1:5
    subplot(6,1,i);
    plot(eeg_A_1(1:5000,i));
end
subplot(6,1,1);
title('raw EEG and EOG');
subplot(6,1,6);
plot(eog_A_1(1:5000,:));

figure;%滤波后的数据
for i=1:5
    subplot(6,1,i);
    plot(Psych_A_1_first(1:5000,i));
end
subplot(6,1,1);
title('EEG and EOG after filted');
subplot(6,1,6);
plot(Psych_A_1_first(1:5000,16));

figure;%用作论文中的图
subplot(4,3,1);
plot(eeg_A_1(1:5000,1));
subplot(4,3,4);
plot(eeg_A_1(1:5000,2));
subplot(4,3,7);
plot(eeg_A_1(1:5000,3));
subplot(4,3,10);
plot(eog_A_1(1:5000,:));
subplot(4,3,2);
plot(Psych_A_1_first(1:5000,1));
subplot(4,3,5);
plot(Psych_A_1_first(1:5000,2));
subplot(4,3,8);
plot(Psych_A_1_first(1:5000,3));
subplot(4,3,11);
plot(Psych_A_1_first(1:5000,16));
%--------------------------------------------------------------------------


%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%可视化前2秒的数据----------------------------------------------------------
subplot(4,3,3);
plot(filted_eeg1(1:5000,1));
subplot(4,3,6);
plot(filted_eeg1(1:5000,2));
subplot(4,3,9);
plot(filted_eeg1(1:5000,3));
subplot(4,3,12);
plot(Psych_A_1_first(1:5000,16));

figure;%滤波后去眼电的5导EEG数据
for i=1:5
    subplot(6,1,i);
    plot(filted_eeg1(1:5000,i));
end
subplot(6,1,1);
title('EEG after EOG removal');
subplot(6,1,6);
plot(Psych_A_1_first(1:5000,16));
%--------------------------------------------------------------------------

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
%clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

subplot(4,1,1)
plot(Psych_A_1_first(1:5000,17));
title('(a)ECG') 
xlabel('Sample index (500 Hz)')
ylabel('uV')
subplot(4,1,2)
plot(rr_interval(1:10));
title('(b)BI') 
xlabel('Time index (1 s)')
ylabel('s')
subplot(4,1,3)
plot(hr(1:10));
title('(c)HR') 
xlabel('Time index (1 s)')
ylabel('Times/s')
subplot(4,1,4)
plot(hrv(1:10));
title('(d)HRV') 
xlabel('Time index (1 s)')


feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_1=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_1 OFS_A_first_1
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_02_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_02_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_02_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_2=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_2 OFS_A_first_2
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_03_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_03_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_03_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_3=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_3 OFS_A_first_3
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_04_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_04_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_04_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_4=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_4 OFS_A_first_4
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_05_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_05_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_05_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
%采用均值补零
%[p q]=find(hr==0);
%for i0=1:length(p)
%    hr(p(i0))=mean(hr);
%end
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%采用均值补零
%[p q]=find(rr_interval==0);
%for i0=1:length(p)
%    rr_interval(p(i0))=mean(rr_interval);
%end
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_5=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_5 OFS_A_first_5
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_06_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_06_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_06_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_6=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_6 OFS_A_first_6
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_07_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_07_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_07_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_7=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_7 OFS_A_first_7
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_08_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_08_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_08_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_8=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_8 OFS_A_first_8
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_09_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_09_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_09_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_9=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_9 OFS_A_first_9
%--------------------------------------------------------------------------


clc;
clear;

%载入数据
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_10_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_10_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_10_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%选择数据窗口的长度为10s,输出低通滤波后的电生理信号
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%采用相关法去眼电
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%将去眼电后的6导EEG替换滤波后的数据
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%采用离散傅立叶变换计算EEG,EOG的功率特征
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%删除多余变量
clear eeg_A_1 eog_A_1 ecg_A_1

%脑电功率的离群点剔除
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%计算瞬时心率和RR间期
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%将异搏点代替为正常值
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%计算心率瞬时变化
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_10=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_10 OFS_A_first_10
%--------------------------------------------------------------------------