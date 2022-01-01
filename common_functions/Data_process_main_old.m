%���������ڶ���ʵ���ȡ��������������ݽ���Ԥ����
%��д������
clc;
clear;
close all;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_01_all'
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(all_data,[],[],10);
%����EOG=���۵�-���۵�Ϊ��12��,��ECG��Ϊ��13��,�Ƴ���14��
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%���ӻ�ǰ10�������----------------------------------------------------------
figure;%�˲�ǰ������
for i=1:5
    subplot(6,1,i);
    plot(all_data(1:5000,i));
end
subplot(6,1,1);
title('raw EEG and EOG');
subplot(6,1,6);
plot(all_data(1:5000,13)-all_data(1:5000,12));

figure;%�˲��������
for i=1:5
    subplot(6,1,i);
    plot(Psych(1:5000,i));
end
subplot(6,1,1);
title('EEG and EOG after filted');
subplot(6,1,6);
plot(Psych(1:5000,12));
%--------------------------------------------------------------------------

%������ط�ȥ�۵�
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%���ӻ�ǰ2�������----------------------------------------------------------
figure;%�˲���ȥ�۵��5��EEG����
for i=1:5
    subplot(6,1,i);
    plot(filted_eeg(1:5000,i));
end
subplot(6,1,1);
title('EEG after EOG removal');
subplot(6,1,6);
plot(Psych(1:5000,12));
%--------------------------------------------------------------------------

%��ȥ�۵���6��EEG�滻�˲��������
Psych(:,1:6)=filted_eeg;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%ɾ���������
clear all_data

%�Ե��۵繦�ʵ���Ⱥ���޳�
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%�������������
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%������
old_OFS_A_second_1=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_1 old_OFS_A_second_1
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_02_all'
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(all_data,[],[],10);
%����EOG=���۵�-���۵�Ϊ��12��,��ECG��Ϊ��13��,�Ƴ���14��
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%������ط�ȥ�۵�
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych(:,1:6)=filted_eeg;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%ɾ���������
clear all_data

%�Ե��۵繦�ʵ���Ⱥ���޳�
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%�������������
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%������
old_OFS_A_second_2=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_2 old_OFS_A_second_2
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_03_all'
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(all_data,[],[],10);
%����EOG=���۵�-���۵�Ϊ��12��,��ECG��Ϊ��13��,�Ƴ���14��
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%������ط�ȥ�۵�
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych(:,1:6)=filted_eeg;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%ɾ���������
clear all_data

%�Ե��۵繦�ʵ���Ⱥ���޳�
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%�������������
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%������
old_OFS_A_second_3=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_3 old_OFS_A_second_3
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_04_all'
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(all_data,[],[],10);
%����EOG=���۵�-���۵�Ϊ��12��,��ECG��Ϊ��13��,�Ƴ���14��
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%������ط�ȥ�۵�
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych(:,1:6)=filted_eeg;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%ɾ���������
clear all_data

%�Ե��۵繦�ʵ���Ⱥ���޳�
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%�������������
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%������
old_OFS_A_second_4=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_4 old_OFS_A_second_4
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_05_all'
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(all_data,[],[],10);
%����EOG=���۵�-���۵�Ϊ��12��,��ECG��Ϊ��13��,�Ƴ���14��
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%������ط�ȥ�۵�
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych(:,1:6)=filted_eeg;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%ɾ���������
clear all_data

%�Ե��۵繦�ʵ���Ⱥ���޳�
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%�������������
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%������
old_OFS_A_second_5=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_5 old_OFS_A_second_5
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_06_all'
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(all_data,[],[],10);
%����EOG=���۵�-���۵�Ϊ��12��,��ECG��Ϊ��13��,�Ƴ���14��
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%������ط�ȥ�۵�
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych(:,1:6)=filted_eeg;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%ɾ���������
clear all_data

%�Ե��۵繦�ʵ���Ⱥ���޳�
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%�������������
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%������
old_OFS_A_second_6=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_6 old_OFS_A_second_6
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_07_all'
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(all_data,[],[],10);
%����EOG=���۵�-���۵�Ϊ��12��,��ECG��Ϊ��13��,�Ƴ���14��
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%������ط�ȥ�۵�
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych(:,1:6)=filted_eeg;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%ɾ���������
clear all_data

%�Ե��۵繦�ʵ���Ⱥ���޳�
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%�������������
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%������
old_OFS_A_second_7=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_7 old_OFS_A_second_7
%--------------------------------------------------------------------------


clc;
clear;

%��������
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_08_all'
%��һ���׶ε����ݽ����˲�
Psych=Data_process_filter(all_data,[],[],10);
%����EOG=���۵�-���۵�Ϊ��12��,��ECG��Ϊ��13��,�Ƴ���14��
Psych(:,12)=Psych(:,13)-Psych(:,12);
Psych(:,13)=Psych(:,14);
Psych(:,14)=[];

%������ط�ȥ�۵�
filted_eeg=Data_process_eogremove(Psych(:,1:6),Psych(:,12),10);

%��ȥ�۵���6��EEG�滻�˲��������
Psych(:,1:6)=filted_eeg;

%������ɢ����Ҷ�任����EEG,EOG�Ĺ�������
eeg_eog_feature=Data_process_FFT(Psych(:,1:12));
%ɾ���������
clear all_data

%�Ե��۵繦�ʵ���Ⱥ���޳�
for i=1:480
    eeg_eog_feature_new(:,i)=Data_process_outlier_hr(eeg_eog_feature(:,i),3);
end

%����˲ʱ���ʺ�RR����
[hr,rr_interval]=Data_process_hr(Psych(:,13));

%���첫�����Ϊ����ֵ
[hr]=Data_process_outlier_hr(hr,3);
[hr]=Data_process_outlier_hr(hr,3);
[rr_interval]=Data_process_outlier_hr(rr_interval,3);
%��������˲ʱ�仯
hrv=diff(rr_interval);
hrv=[hrv;hrv(length(hrv),:)];
hrv=abs(hrv);

%�������������
feature_all=[eeg_eog_feature_new hr rr_interval hrv];
%������
old_OFS_A_second_8=feature_all;

save D:\matlab\matlab7\work\OFS_data_process\old_OFS_feature\old_OFS_A_second_8 old_OFS_A_second_8
%--------------------------------------------------------------------------