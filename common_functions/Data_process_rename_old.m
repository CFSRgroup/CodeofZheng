%本程序用于对老实验截取后的心理生理数据进行重命名和转换格式
%编写者尹钟
clc;
clear;
%--------------------------------------------------------------------------
%载入文件
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\eeg1.txt';
%重命名
all_data=eeg1;
%另存为mat文件
save D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_01_all all_data
length(all_data)/500
clear;
%--------------------------------------------------------------------------
%载入文件
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\eeg2.txt';
%重命名
all_data=eeg2;
%另存为mat文件
save D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_02_all all_data
length(all_data)/500
clear;
%--------------------------------------------------------------------------
%载入文件
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\eeg3.txt';
%重命名
all_data=eeg3;
%另存为mat文件
save D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_03_all all_data
length(all_data)/500
clear;
%--------------------------------------------------------------------------
%载入文件
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\eeg4.txt';
%重命名
all_data=eeg4;
%另存为mat文件
save D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_04_all all_data
length(all_data)/500
clear;
%--------------------------------------------------------------------------
%载入文件
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\eeg5.txt';
%重命名
all_data=eeg5;
%另存为mat文件
save D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_05_all all_data
length(all_data)/500
clear;
%--------------------------------------------------------------------------
%载入文件
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\eeg6.txt';
%重命名
all_data=eeg6;
%另存为mat文件
save D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_06_all all_data
length(all_data)/500
clear;
%--------------------------------------------------------------------------
%载入文件
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\eeg7.txt';
%重命名
all_data=eeg7;
%另存为mat文件
save D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_07_all all_data
length(all_data)/500
clear;
%--------------------------------------------------------------------------
%载入文件
load 'D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\eeg8.txt';
%重命名
all_data=eeg8;
%另存为mat文件
save D:\matlab\matlab7\work\OFS_data_process\old_OFS_A\OFS_liujinqing_second_08_all all_data
length(all_data)/500
clear;