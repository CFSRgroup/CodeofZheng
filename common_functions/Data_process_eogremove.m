function [eeg_noeogdata]=Data_process_eogremove(eeg,eog,lseg)
%本函数用于EEG中EOG伪迹的去除
%编写者020100078尹钟

%测量输入数据的长度
ldata=length(eeg(:,1));
%选取的数据窗长度,默认采样频率为500HZ
lwin=lseg*500;
%计算数据窗的个数
nwin=ldata/lwin;
%设置采样频率为500Hz
fs=500;

%依据确定的数据窗依次对数据滤波
for di=1:nwin
    eogdata=eog(lwin*(di-1)+1:lwin*di,1);
    eegdata=eeg(lwin*(di-1)+1:lwin*di,:);
    %采用butter滤波器对眼电进一步滤波
    [n_piont n_channel]=size(eegdata);
    %采用3阶滤波器
    filterorder=3;
    %带通设置,值得注意的是这里的参数时归一化后的,1为奈奎斯特频率,即采样频率的一半
    %滤波的目的是去除眼电中的alpha节律与beta节律成分
    filtercutoff=[2*8/fs];
    %计算滤波器参数,值得注意的是此处采用了低通滤波器
    [f_b, f_a] = butter(filterorder,filtercutoff,'low');
    %对数据进行带通滤波
    filted_eogdata(:,:) = filtfilt(f_b,f_a,eogdata);%也可以用函数filter
    %接下来采用相关法去除眼电伪迹，计算eog与每一导eeg的方差与协方差
    for j = 1:n_channel
        cov_eog_eeg=cov(filted_eogdata(:,1),eegdata(:,j));
        %计算传递系数
        b=cov_eog_eeg(1,2)/cov_eog_eeg(1,1);
        eegdata_new(:,j)=eegdata(:,j)-b*filted_eogdata;
    end
    eeg_noeogdata(lwin*(di-1)+1:lwin*di,:)=eegdata_new;
end