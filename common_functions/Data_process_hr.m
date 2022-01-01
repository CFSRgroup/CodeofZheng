function [hr,rr_interval]=Data_process_hr(ecg_data)
%���������ڼ���˲ʱ���ʺ�RR_interval
%�������ݸ�ʽΪ������
%���ΪHR��RR_interval
%��д��020100078����
%ѡ��ʱ�䳤��Ϊ4��
lseg=4;
%ѡȡ�����ݴ�����,Ĭ�ϲ���Ƶ��Ϊ500HZ
lwin=lseg*500;
%�����һ��11���ڵ�����-----------------------------------------------------
%�������ݵĳ���,���������ݰ���������
datatime=length(ecg_data)/500;
rr_interval=zeros(datatime,1);
%���ò���Ƶ��
sr=500;
for i=1:(datatime-5)
    y=ecg_data(sr*(i-1)+1:sr*(i-1)+lwin);
    %�����ĵ��źŵ�һ�ײ��
    y1=diff(y);
    %����ά��һ��
    y1=[y1;y1(length(y1))];
    %�����ĵ��źŵĶ��ײ��
    y2=diff(y1);
    %����ά��һ��
    y2=[y2;y2(length(y2))];
    %�����ĵ��źŵ�һ�ײ������ײ�ֵ��ۼ�,Ϊ�˷�����R��
    y=1.3*y1+1.1*y2;
    y=y1;
    %Ϊ�˼��QRS���ϲ���λ��,����һ����ֵ
    th=max((y));
    th=th*8/9;
    c=0;
    %���QRS���ϲ���λ��
    for m=1:1:lwin
        if (y(m))>th
           qrs(m)=1;
           c=c+1;
           m=m+15;
        else
          qrs(m)=0;
        end
    end
    %�Ը���������
    qrs_diff=diff(qrs);
    qrs_diff=[qrs_diff qrs_diff(1,length(qrs_diff))];
    %��������R����ʼλ�õ�����
    [a b]=find(qrs_diff>0);
    %���R������(��λ��)
    if numel(b)==1 | numel(b)==0
        rr_interval_sec=mean(rr_interval);
    else
        rr_interval_sec=(b(2)-b(1))*0.002;
    end
    %���˲ʱ����
    hr_sec=60/rr_interval_sec;
    %����HR��R������
    rr_interval(i,:)=rr_interval_sec;
    hr(i,:)=hr_sec;
end
%��ȫ��������ӵ�����
hr(datatime-3:datatime,1)=hr(datatime-6,1);
rr_interval(datatime-3:datatime,1)=rr_interval(datatime-6,1);