function [eeg_noeogdata]=Data_process_eogremove(eeg,eog,lseg)
%����������EEG��EOGα����ȥ��
%��д��020100078����

%�����������ݵĳ���
ldata=length(eeg(:,1));
%ѡȡ�����ݴ�����,Ĭ�ϲ���Ƶ��Ϊ500HZ
lwin=lseg*500;
%�������ݴ��ĸ���
nwin=ldata/lwin;
%���ò���Ƶ��Ϊ500Hz
fs=500;

%����ȷ�������ݴ����ζ������˲�
for di=1:nwin
    eogdata=eog(lwin*(di-1)+1:lwin*di,1);
    eegdata=eeg(lwin*(di-1)+1:lwin*di,:);
    %����butter�˲������۵��һ���˲�
    [n_piont n_channel]=size(eegdata);
    %����3���˲���
    filterorder=3;
    %��ͨ����,ֵ��ע���������Ĳ���ʱ��һ�����,1Ϊ�ο�˹��Ƶ��,������Ƶ�ʵ�һ��
    %�˲���Ŀ����ȥ���۵��е�alpha������beta���ɳɷ�
    filtercutoff=[2*8/fs];
    %�����˲�������,ֵ��ע����Ǵ˴������˵�ͨ�˲���
    [f_b, f_a] = butter(filterorder,filtercutoff,'low');
    %�����ݽ��д�ͨ�˲�
    filted_eogdata(:,:) = filtfilt(f_b,f_a,eogdata);%Ҳ�����ú���filter
    %������������ط�ȥ���۵�α��������eog��ÿһ��eeg�ķ�����Э����
    for j = 1:n_channel
        cov_eog_eeg=cov(filted_eogdata(:,1),eegdata(:,j));
        %���㴫��ϵ��
        b=cov_eog_eeg(1,2)/cov_eog_eeg(1,1);
        eegdata_new(:,j)=eegdata(:,j)-b*filted_eogdata;
    end
    eeg_noeogdata(lwin*(di-1)+1:lwin*di,:)=eegdata_new;
end