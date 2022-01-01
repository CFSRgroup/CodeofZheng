%本程序用于实验1的EEG特征提取
clc;
clear;
close all;

for pp=1:6
for qq=1:10
clear power feature_f data eeg
eval(['load F:\matlab\trial_procedure\study_1\ex2_filter_ica\session1\s' num2str(pp) '_' num2str(qq)])

%测量每个矩阵的长度
a1=size(eeg,1);
a1=a1/500;

%计算频域特征
for i=1:a1/2
    for j=1:11
        data=eeg(1+(i-1)*1000:i*1000,j);
        %进行傅里叶变换
        Fs=500;%采样频率
        T=1/Fs;%采样周期
        L=1000;%信号长度
        NFFT=2^nextpow2(L);
        psd=fft(data,NFFT)/L;
        psd=2*abs(psd(1:NFFT/2+1));
        f=(Fs/2*linspace(0,1,NFFT/2+1))';
        theta=mean(psd(12:17,1)); %theta频段（4-8Hz)
        alpha=mean(psd(18:27,1));%alpha频段(8-13Hz)
        beta=mean(psd(28:62,1));%beta频段(14-30Hz)
        gamma=mean(psd(63:82,1));%gamma频段(31-40Hz)
        power(1,1+(j-1)*4:j*4)=[theta alpha beta gamma];       
    end
        %大脑右左半球的功率之差
        diff_power_1=power(1,1+(2-1)*4:2*4)-power(1,1+(1-1)*4:1*4);
        diff_power_2=power(1,1+(5-1)*4:5*4)-power(1,1+(4-1)*4:4*4);
        diff_power_3=power(1,1+(8-1)*4:8*4)-power(1,1+(7-1)*4:7*4);
        diff_power_4=power(1,1+(11-1)*4:11*4)-power(1,1+(10-1)*4:10*4);
        diff_power(1,:)=[diff_power_1 diff_power_2 diff_power_3 diff_power_4];
        feature_f(i,:)=[power diff_power];
end

%计算时域特征
for i=1:a1/2
    for j=1:11
        data=eeg(1+(i-1)*1000:i*1000,j);
        eeg_mean(i,j)=mean(data);
        %eeg_var(i,j)=var(data);
        eeg_zcr(i,j)=zcr(data);
        eeg_entropy_shannon(i,j)=wentropy(data,'shannon'); %Entropy (wavelet packet)
        %eeg_entropy_logenergy(i,j)=wentropy(data,'log energy'); %log energy对数能量熵
        %eeg_entropy_spectral(i,j)=FFT_entropy(data);
        eeg_kurtosis(i,j)=kurtosis(data);
        eeg_skewness(i,j)=skewness(data);
        
        eeg_peak(i,j)=max(abs(data));
        eeg_std(i,j)=std(data);
        [a,b]=size(data);
        eeg_rms(i,j)=sqrt(sum(data.^2)./a); %均方根公式，这里是矩阵的分元素乘方，用“.^”
        eeg_sf(i,j)=eeg_rms(i,j)/eeg_mean(i,j);
        eeg_cf(i,j)=eeg_peak(i,j)/eeg_rms(i,j);
        eeg_pi(i,j)=eeg_peak(i,j)/eeg_mean(i,j);
    end
end

feature_t=[eeg_zcr eeg_entropy_shannon eeg_kurtosis eeg_skewness eeg_peak eeg_std eeg_rms eeg_sf eeg_cf eeg_pi];

x=[feature_f feature_t];

eval(['save F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_' num2str(qq) ' x'])
end
end

% clc;
% clear;
% close all;
% 
% for pp=1:6
% for qq=1:10
% clear power feature_f data eeg
% eval(['load F:\matlab\trial_procedure\study_1\ex2_filter_ica\session2\s' num2str(pp) '_' num2str(qq)])
% 
% %测量每个矩阵的长度
% a1=size(eeg,1);
% a1=a1/500;
% 
% %计算频域特征
% for i=1:a1/2
%     for j=1:11
%         data=eeg(1+(i-1)*1000:i*1000,j);
%         %进行傅里叶变换
%         Fs=500;%采样频率
%         T=1/Fs;%采样周期
%         L=1000;%信号长度
%         NFFT=2^nextpow2(L);
%         psd=fft(data,NFFT)/L;
%         psd=2*abs(psd(1:NFFT/2+1));
%         f=(Fs/2*linspace(0,1,NFFT/2+1))';
%         theta=mean(psd(12:17,1)); %theta频段（4-8Hz)
%         alpha=mean(psd(18:27,1));%alpha频段(8-13Hz)
%         beta=mean(psd(28:62,1));%beta频段(14-30Hz)
%         gamma=mean(psd(63:82,1));%gamma频段(31-40Hz)
%         power(1,1+(j-1)*4:j*4)=[theta alpha beta gamma];       
%     end
%         %大脑右左半球的功率之差
%         diff_power_1=power(1,1+(2-1)*4:2*4)-power(1,1+(1-1)*4:1*4);
%         diff_power_2=power(1,1+(5-1)*4:5*4)-power(1,1+(4-1)*4:4*4);
%         diff_power_3=power(1,1+(8-1)*4:8*4)-power(1,1+(7-1)*4:7*4);
%         diff_power_4=power(1,1+(11-1)*4:11*4)-power(1,1+(10-1)*4:10*4);
%         diff_power(1,:)=[diff_power_1 diff_power_2 diff_power_3 diff_power_4];
%         feature_f(i,:)=[power diff_power];
% end
% 
% %计算时域特征
% for i=1:a1/2
%     for j=1:11
%         data=eeg(1+(i-1)*1000:i*1000,j);
%         eeg_mean(i,j)=mean(data);
%         %eeg_var(i,j)=var(data);
%         eeg_zcr(i,j)=zcr(data);
%         eeg_entropy_shannon(i,j)=wentropy(data,'shannon'); %Entropy (wavelet packet)
%         %eeg_entropy_logenergy(i,j)=wentropy(data,'log energy'); %log energy对数能量熵
%         %eeg_entropy_spectral(i,j)=FFT_entropy(data);
%         eeg_kurtosis(i,j)=kurtosis(data);
%         eeg_skewness(i,j)=skewness(data);
%         
%         eeg_peak(i,j)=max(abs(data));
%         eeg_std(i,j)=std(data);
%         [a,b]=size(data);
%         eeg_rms(i,j)=sqrt(sum(data.^2)./a); %均方根公式，这里是矩阵的分元素乘方，用“.^”
%         eeg_sf(i,j)=eeg_rms(i,j)/eeg_mean(i,j);
%         eeg_cf(i,j)=eeg_peak(i,j)/eeg_rms(i,j);
%         eeg_pi(i,j)=eeg_peak(i,j)/eeg_mean(i,j);
%     end
% end
% 
% feature_t=[eeg_zcr eeg_entropy_shannon eeg_kurtosis eeg_skewness eeg_peak eeg_std eeg_rms eeg_sf eeg_cf eeg_pi];
% 
% x=[feature_f feature_t];
% 
% eval(['save F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_' num2str(qq) ' x'])
% end
% end