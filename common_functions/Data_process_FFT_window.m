function [eeg_eog_p]=Data_process_FFT_window(eeg_eog)
%����������EEG��EOG�������ܶȵļ���,���㷽ʽΪ(�����ƶ�����,�ص���Ϊ90%)��ȡһ��FFT����
%�������ݸ�ʽΪʱ������*��
%��д��020100078����

%�����������ݵĳ���
ldata=length(eeg_eog(:,1));
%ѡ��ʱ�䳤��
lseg=10;

%ѡȡ��δ�ص����ݴ�����,Ĭ�ϲ���Ƶ��Ϊ500HZ
lwin=500;

%�������ݴ��ĸ���
nwin=ldata/500-(lseg-1);

for di=1:nwin
    eeg_eog_sec=eeg_eog(lwin*(di-1)+1:lwin*(di+(lseg-1)),:);
    %��EEGEOG���ݽ���FFT
    eeg_eog_sec_p=fft(eeg_eog_sec);
    %���㹦���ף���λ��mV^2)
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

