%这是一个读取ASCII码文件的程序,同时提取了10秒至结束的数据
%编写者020100078 尹钟
clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_01.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_01_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_01_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_01_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_02.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_02_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_02_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_02_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_03.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_03_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_03_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_03_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_04.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_04_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_04_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_04_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_05.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_05_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_05_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_05_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_06.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_06_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_06_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_06_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_07.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_07_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_07_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_07_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_08.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_08_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_08_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_08_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_09.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_09_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_09_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_09_ecg ecg_data
%--------------------------------------------------------------------------

clc;
clear;

%读取指针,第一项为文件名称
fid=fopen('OFS_linwei_second_10.m00','r');

%读取文件头
head_1=fgets(fid);
head_2=fgets(fid);

%读取EEG、EOG、ECG
main_data=fscanf(fid,'%f',[17 inf]);
fclose(fid);
main_data=main_data';

%截取数据
main_data=main_data([5001:150000],:);
%提取EEG数据
eeg_data=main_data(:,[1:15]);
%提取EOG数据
eog_data=main_data(:,16);
%提取ECG数据
ecg_data=main_data(:,17);

%保存截取后的数据
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_10_eeg eeg_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_10_eog eog_data
save D:\matlab\matlab7\work\OFS_data_process\OFS_linwei_second_10_ecg ecg_data