function [all_filterdata]=Data_process_filter(eeg,eog,ecg,lseg);
%����������EEG,EOG,ECG���ݵ��˲�,���д�ͨ����Ϊ0.5-40HZ

%�����������ݵĳ���
ldata=length(eeg(:,1));
%ѡȡ�����ݴ�����,Ĭ�ϲ���Ƶ��Ϊ500HZ
lwin=lseg*500;
%�������ݴ��ĸ���
nwin=ldata/lwin;
%�ϲ�EEG,ECG,EOG����
choosedata(:,:)=[eeg eog ecg];

%����ȷ�������ݴ����ζ������˲�
for di=1:nwin
    data(:,:)=choosedata(lwin*(di-1)+1:lwin*di,:);
    %����butter�˲����˲�
    [n_piont n_channel]=size(data);
    %���ò���Ƶ��Ϊ500Hz
    fs=500;
    %����3���˲���
    filterorder=3;
    %��ͨ����,ֵ��ע���������Ĳ���ʱ��һ�����,1Ϊ�ο�˹��Ƶ��,������Ƶ�ʵ�һ��
    filtercutoff=[2*40/fs];
    %�����˲�������,ֵ��ע����Ǵ˴������˵�ͨ�˲���
    [f_b, f_a] = butter(filterorder,filtercutoff,'low');
    %�����ݽ��д�ͨ�˲�
    j=1;
    for j = 1:n_channel
        filterdata(:,j) = filtfilt(f_b,f_a,data(:,j));%Ҳ�����ú���filter
    end
    %figure;
    %subplot 211; plot(data(:,1));
    %subplot 212; plot(filterdata(:,1));
    all_filterdata(lwin*(di-1)+1:lwin*di,:)=filterdata;
end