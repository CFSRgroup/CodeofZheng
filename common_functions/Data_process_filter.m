function [all_filterdata]=Data_process_filter(eeg,eog,ecg,lseg);
%本函数用于EEG,EOG,ECG数据的滤波,其中带通设置为0.5-40HZ

%测量输入数据的长度
ldata=length(eeg(:,1));
%选取的数据窗长度,默认采样频率为500HZ
lwin=lseg*500;
%计算数据窗的个数
nwin=ldata/lwin;
%合并EEG,ECG,EOG数据
choosedata(:,:)=[eeg eog ecg];

%依据确定的数据窗依次对数据滤波
for di=1:nwin
    data(:,:)=choosedata(lwin*(di-1)+1:lwin*di,:);
    %采用butter滤波器滤波
    [n_piont n_channel]=size(data);
    %设置采样频率为500Hz
    fs=500;
    %采用3阶滤波器
    filterorder=3;
    %带通设置,值得注意的是这里的参数时归一化后的,1为奈奎斯特频率,即采样频率的一半
    filtercutoff=[2*40/fs];
    %计算滤波器参数,值得注意的是此处采用了低通滤波器
    [f_b, f_a] = butter(filterorder,filtercutoff,'low');
    %对数据进行带通滤波
    j=1;
    for j = 1:n_channel
        filterdata(:,j) = filtfilt(f_b,f_a,data(:,j));%也可以用函数filter
    end
    %figure;
    %subplot 211; plot(data(:,1));
    %subplot 212; plot(filterdata(:,1));
    all_filterdata(lwin*(di-1)+1:lwin*di,:)=filterdata;
end