%����һ����ȡASCII���ļ��ĳ���,ͬʱ��ȡ��10��������������
%��д��020100078 ����
clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_01.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_01_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_01_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_01_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_02.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_02_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_02_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_02_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_03.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_03_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_03_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_03_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_04.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_04_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_04_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_04_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_05.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_05_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_05_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_05_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_06.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_06_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_06_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_06_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_07.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_07_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_07_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_07_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_08.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_08_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_08_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_08_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_09.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_09_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_09_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_09_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%��ȡָ��,��һ��Ϊ�ļ�����
fid=fopen('OFS_linwei_second_10.m00','r');

%��ȡ�ļ�ͷ
head_1=fgets(fid);
head_2=fgets(fid);

%��ȡEEG��EOG��ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%��ȡ����
main_data=main_data([5001:150000],:);
%��ȡEEG����
eeg_data=main_data(:,[1:15]);
%��ȡEOG����
eog_data=main_data(:,16);
%��ȡECG����
ecg_data=main_data(:,17);

%�����ȡ�������
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_10_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_10_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_10_ecg ecg_data