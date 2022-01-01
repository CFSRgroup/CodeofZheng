function [eeg_eog_p]=Data_process_FFT_window(eeg_eog)
%本函数用于EEG与EOG功率谱密度的计算,计算方式为(基于移动窗口,重叠率为90%)提取一次FFT参数
%输入数据格式为时间序列*导
%编写者020100078尹钟

%测量输入数据的长度
ldata=length(eeg_eog(:,1));
%选择时间长度
lseg=10;

%选取的未重叠数据窗长度,默认采样频率为500HZ
lwin=500;

%计算数据窗的个数
nwin=ldata/500-(lseg-1);

for di=1:nwin
    eeg_eog_sec=eeg_eog(lwin*(di-1)+1:lwin*(di+(lseg-1)),:);
    %对EEGEOG数据进行FFT
    eeg_eog_sec_p=fft(eeg_eog_sec);
    %计算功率谱，单位（mV^2)
    eeg_eog_sec_p=eeg_eog_sec_p.*conj(eeg_eog_sec_p)/1000000;
    eeg_eog_sec_p=eeg_eog_sec_p(1:40,:);
    [a,b]=size(eeg_eog_sec_p);
    eeg_eog_sec_p=reshape(eeg_eog_sec_p,a*b,1);
    eeg_eog_sec_p=eeg_eog_sec_p';
    eeg_eog_p(di,:)=eeg_eog_sec_p;
end

eeg_eog_mean=mean(eeg_eog_p);

for di=nwin+1:ldata/500
    eeg_eog_p(di,:)=eeg_eog_mean;
end

