function [hr,rr_interval]=Data_process_hr(ecg_data)
%本函数用于计算瞬时心率和RR_interval
%输入数据格式为列向量
%输出为HR和RR_interval
%编写者020100078尹钟
%选择时间长度为4秒
lseg=4;
%选取的数据窗长度,默认采样频率为500HZ
lwin=lseg*500;
%计算第一个11秒内的心率-----------------------------------------------------
%测量数据的长度,并计算数据包含几秒钟
datatime=length(ecg_data)/500;
rr_interval=zeros(datatime,1);
%设置采样频率
sr=500;
for i=1:(datatime-5)
    y=ecg_data(sr*(i-1)+1:sr*(i-1)+lwin);
    %计算心电信号的一阶差分
    y1=diff(y);
    %保持维数一致
    y1=[y1;y1(length(y1))];
    %计算心电信号的二阶差分
    y2=diff(y1);
    %保持维数一致
    y2=[y2;y2(length(y2))];
    %计算心电信号的一阶差分与二阶差分的累计,为了方便检测R波
    y=1.3*y1+1.1*y2;
    y=y1;
    %为了检测QRS复合波的位置,设置一个阈值
    th=max((y));
    th=th*8/9;
    c=0;
    %检测QRS复合波的位置
    for m=1:1:lwin
        if (y(m))>th
           qrs(m)=1;
           c=c+1;
           m=m+15;
        else
          qrs(m)=0;
        end
    end
    %对该序列求差分
    qrs_diff=diff(qrs);
    qrs_diff=[qrs_diff qrs_diff(1,length(qrs_diff))];
    %计算两个R波开始位置的索引
    [a b]=find(qrs_diff>0);
    %求得R波间期(单位秒)
    if numel(b)==1 | numel(b)==0
        rr_interval_sec=mean(rr_interval);
    else
        rr_interval_sec=(b(2)-b(1))*0.002;
    end
    %求得瞬时心率
    hr_sec=60/rr_interval_sec;
    %保存HR和R波间期
    rr_interval(i,:)=rr_interval_sec;
    hr(i,:)=hr_sec;
end
%补全最后三秒钟的数据
hr(datatime-3:datatime,1)=hr(datatime-6,1);
rr_interval(datatime-3:datatime,1)=rr_interval(datatime-6,1);