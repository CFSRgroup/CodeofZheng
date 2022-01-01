function [eeg_eog_p]=Data_process_FFT(eeg_eog)
%����������EEG��EOG�������ܶȵļ���,���㷽ʽΪһ������ȡһ��FFT����
%�������ݸ�ʽΪʱ������*��
%��д��020100078����

%�����������ݵĳ���
ldata=length(eeg_eog(:,1));
%ѡ��ʱ�䳤��Ϊ1��
lseg=1;
%ѡȡ�����ݴ�����,Ĭ�ϲ���Ƶ��Ϊ500HZ
lwin=lseg*500;
%�������ݴ��ĸ���
nwin=ldata/lwin;
%��ÿһ��Ƭ�Ͻ���FFT
for di=1:nwin
    eeg_eog_sec=eeg_eog(lwin*(di-1)+1:lwin*di,:);
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









