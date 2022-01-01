clc;
clear;
close all;
warning off;

% T1：测试管理平台开发项目组
% T2：公司项目组
% T3：技术测试组
% T4：金融市场业务项目组
% T5：理财子公司项目组
% T6：零售项目组
% T7：同业项目组
% T8：托管项目组
% T9：综合二组
% T10：综合一组

T1 = [0.9600 0.9600 0.8800 0.8800 0.3333 0.5625];
T2 = [0.9676 0.9707 0.9651 0.9730 0.4324 0.4567];
T3 = [0.7750 0.7750 0.7750 0.7750 0.1071 0.1000];
T4 = [1.000 1.000 1.000 1.000 0.7500 0.9091];
T5 = [0.9937 0.9968 0.9968 0.9746 0.4381 0.4367];
T6 = [0.9536 0.9668 0.9461 0.9469 0.6923 0.7435];
T7 = [0.9438 0.9389 0.9497 0.9334 0.5236 0.5926];
T8 = [0.9973 0.9920 0.9987 0.9987 0.8667 0.8550];
T9= [0.6377 0.8591 0.8536 0.8230 0.4051 0.4573];
T10 = [0.9829 0.9772 0.9730 0.9761 0.6681 0.6486];

acc_T1 = mean(T1);
sd_T1 = std(T1);
acc_T2 = mean(T2);
sd_T2 = std(T2);
acc_T3 = mean(T3);
sd_T3 = std(T3);
acc_T4 = mean(T4);
sd_T4 = std(T4);
acc_T5 = mean(T5);
sd_T5 = std(T5);
acc_T6 = mean(T6);
sd_T6 = std(T6);
acc_T7 = mean(T7);
sd_T7 = std(T7);
acc_T8 = mean(T8);
sd_T8 = std(T8);
acc_T9 = mean(T9);
sd_T9 = std(T9);
acc_T10 = mean(T10);
sd_T10 = std(T10);

acc = [acc_T1 acc_T2 acc_T3 acc_T4 acc_T5 acc_T6 acc_T7 acc_T8 acc_T9 acc_T10];
sd = [sd_T1 sd_T2 sd_T3 sd_T4 sd_T5 sd_T6 sd_T7 sd_T8 sd_T9 sd_T10];

bar(1,acc(1,1));hold on;
set(gca,'XTick',1:1:10); 
set(gca,'XTickLabel',{'测试管理平台开发项目组','公司项目组','技术测试组','金融市场业务项目组','理财子公司项目组','零售项目组','同业项目组','托管项目组','综合二组','综合一组'});
bar(2,acc(1,2));hold on;
bar(3,acc(1,3));hold on;
bar(4,acc(1,4));hold on;
bar(5,acc(1,5));hold on;
bar(6,acc(1,6));hold on;
bar(7,acc(1,7));hold on;
bar(8,acc(1,8));hold on;
bar(9,acc(1,9));hold on;
bar(10,acc(1,10));hold on;
xtickangle(50);
% set(gca,'YTick',0:0.05:0.5);
ylabel('jmeter执行通过率','FontWeight','bold');
title('误差棒图','FontWeight','bold');
grid on;
hold on;
x = 1:1:10;
y=acc(1,:);
err=sd(1,:);
errorbar(x,y,err);

% bar(acc,'DisplayName','acc');
% title('误差棒图','FontWeight','bold');
% grid on;
% hold on;
% set(gca,'XTickLabel',{'测试管理平台开发项目组','公司项目组','技术测试组','金融市场业务项目组','理财子公司项目组','零售项目组','同业项目组','托管项目组','综合二组','综合一组'},'FontWeight','bold');
% xtickangle(50);
% errorbar(acc,sd); 
% ylabel('jmeter执行通过率','FontWeight','bold');
