%��������OFS���ݴ����������
%��д������
clc;
clear;
close all;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_01_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_01_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_01_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%���ӻ�ǰ10�������----------------------------------------------------------
figure;%�˲�ǰ������
for i=1:5
    subplot(6,1,i);
    plot(eeg_A_1(1:5000,i));
end
subplot(6,1,1);
title('raw EEG and EOG');
subplot(6,1,6);
plot(eog_A_1(1:5000,:));

figure;%�˲��������
for i=1:5
    subplot(6,1,i);
    plot(Psych_A_1_first(1:5000,i));
end
subplot(6,1,1);
title('EEG and EOG after filted');
subplot(6,1,6);
plot(Psych_A_1_first(1:5000,16));

figure;%���������е�ͼ
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


%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%���ӻ�ǰ2�������----------------------------------------------------------
subplot(4,3,3);
plot(filted_eeg1(1:5000,1));
subplot(4,3,6);
plot(filted_eeg1(1:5000,2));
subplot(4,3,9);
plot(filted_eeg1(1:5000,3));
subplot(4,3,12);
plot(Psych_A_1_first(1:5000,16));

figure;%�˲���ȥ�۵��5��EEG����
for i=1:5
    subplot(6,1,i);
    plot(filted_eeg1(1:5000,i));
end
subplot(6,1,1);
title('EEG after EOG removal');
subplot(6,1,6);
plot(Psych_A_1_first(1:5000,16));
%--------------------------------------------------------------------------

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
%clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
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

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_02_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_02_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_02_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_2=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_2 OFS_A_first_2
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_03_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_03_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_03_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_3=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_3 OFS_A_first_3
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_04_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_04_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_04_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_4=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_4 OFS_A_first_4
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_05_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_05_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_05_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
%���þ�ֵ����
%[p q]=find(hr==0);
%for i0=1:length(p)
%    hr(p(i0))=mean(hr);
%end
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%���þ�ֵ����
%[p q]=find(rr_interval==0);
%for i0=1:length(p)
%    rr_interval(p(i0))=mean(rr_interval);
%end
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_5=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_5 OFS_A_first_5
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_06_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_06_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_06_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_6=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_6 OFS_A_first_6
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_07_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_07_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_07_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_7=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_7 OFS_A_first_7
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_08_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_08_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_08_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_8=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_8 OFS_A_first_8
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_09_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_09_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_09_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_9=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_9 OFS_A_first_9
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_10_eeg'
eeg_A_1=eeg_data;
clear eeg_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_10_eog'
eog_A_1=eog_data;
clear eog_data;

load 'D:\matlab\matlab7\work\OFS_data_process\OFS_A\OFS_jiangqiang_first_10_ecg'
ecg_A_1=ecg_data;
clear ecg_data;

%ѡ�����ݴ��ڵĳ���Ϊ10s,�����ͨ�˲���ĵ������ź�
Psych_A_1_first=Data_process_filter(eeg_A_1,eog_A_1,ecg_A_1,10);

%������ط�ȥ�۵�
filted_eeg1=Data_process_eogremove(Psych_A_1_first(:,1:6),Psych_A_1_first(:,16),10);
filted_eeg2=Data_process_eogremove(Psych_A_1_first(:,13),Psych_A_1_first(:,16),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych_A_1_first(:,1:6)=filted_eeg1;
Psych_A_1_first(:,13)=filted_eeg2;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych_A_1_first(:,1:16));
%ɾ���������
clear eeg_A_1 eog_A_1 ecg_A_1

%�Ե繦�ʵ���Ⱥ���޳�
for i=1:640
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych_A_1_first(:,17));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

feature_all=[eeg_eog_feature_new hr rr_interval hrv];
OFS_A_first_10=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\OFS_feature\OFS_A_first_10 OFS_A_first_10
%--------------------------------------------------------------------------