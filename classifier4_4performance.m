clc;
clear;
close all;
warning off;

%sae
load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task1_session1
load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task1_session2
load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task2_session2

sae_case1=(per_index_single1+per_index_single2)./2;
sae_case2=(per_index_single3+per_index_single4)./2;
sae_case1_mean=mean(sae_case1);
sae_case1_std=std(sae_case1);
sae_case2_mean=mean(sae_case2);
sae_case2_std=std(sae_case2);
sae_case1=[sae_case1_mean; sae_case1_std];
sae_case2=[sae_case2_mean; sae_case2_std];

%dbn
load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task1_session1
load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task1_session2
load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task2_session2

dbn_case1=(per_index_single1+per_index_single2)./2;
dbn_case2=(per_index_single3+per_index_single4)./2;
dbn_case1_mean=mean(dbn_case1);
dbn_case1_std=std(dbn_case1);
dbn_case2_mean=mean(dbn_case2);
dbn_case2_std=std(dbn_case2);
dbn_case1=[dbn_case1_mean; dbn_case1_std];
dbn_case2=[dbn_case2_mean; dbn_case2_std];

%cnn
load F:\matlab\trial_procedure\study_1\data_analysis\cnn_cross_subject\task1_session1
load F:\matlab\trial_procedure\study_1\data_analysis\cnn_cross_subject\task1_session2
load F:\matlab\trial_procedure\study_1\data_analysis\cnn_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\cnn_cross_subject\task2_session2

cnn_case1=(per_index_single1+per_index_single2)./2;
cnn_case2=(per_index_single3+per_index_single4)./2;
cnn_case1_mean=mean(cnn_case1);
cnn_case1_std=std(cnn_case1);
cnn_case2_mean=mean(cnn_case2);
cnn_case2_std=std(cnn_case2);
cnn_case1=[cnn_case1_mean; cnn_case1_std];
cnn_case2=[cnn_case2_mean; cnn_case2_std];

save F:\matlab\trial_procedure\study_1\classifier4_4performance\data1 sae_case1 dbn_case1 cnn_case1
save F:\matlab\trial_procedure\study_1\classifier4_4performance\data2 sae_case2 dbn_case2 cnn_case2

%%case1
%accuracy
subplot(2,4,1);
bar(1,sae_case1(1,4),'b')
title('(a)','FontWeight','bold');
set(gca,'XTick',1:1:3);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'SAE','DBN','CNN'});
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
bar(2,dbn_case1(1,4),'m')
hold on;
bar(3,cnn_case1(1,4),'g')
x = 1:1:3;
y=[sae_case1(1,4) dbn_case1(1,4) cnn_case1(1,4)];
err=[sae_case1(2,4) dbn_case1(2,4) cnn_case1(2,4)];
errorbar(x,y,err);

%sensitivity
subplot(2,4,2);
bar(1,sae_case1(1,6),'b')
title('(b)','FontWeight','bold');
set(gca,'XTick',1:1:3);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'SAE','DBN','CNN'});
ylabel('Sensitivity','FontWeight','bold');
grid on;
hold on;
bar(2,dbn_case1(1,6),'m')
hold on;
bar(3,cnn_case1(1,6),'g')
x = 1:1:3;
y=[sae_case1(1,6) dbn_case1(1,6) cnn_case1(1,6)];
err=[sae_case1(2,6) dbn_case1(2,6) cnn_case1(2,6)];
errorbar(x,y,err);

%spe
subplot(2,4,3);
bar(1,sae_case1(1,7),'b')
title('(c)','FontWeight','bold');
set(gca,'XTick',1:1:3);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'SAE','DBN','CNN'});
ylabel('Specificity','FontWeight','bold');
grid on;
hold on;
bar(2,dbn_case1(1,7),'m')
hold on;
bar(3,cnn_case1(1,7),'g')
x = 1:1:3;
y=[sae_case1(1,7) dbn_case1(1,7) cnn_case1(1,7)];
err=[sae_case1(2,7) dbn_case1(2,7) cnn_case1(2,7)];
errorbar(x,y,err);

%f1
subplot(2,4,4);
bar(1,sae_case1(1,10),'b')
title('(d)','FontWeight','bold');
set(gca,'XTick',1:1:3);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'SAE','DBN','CNN'});
ylabel('F1-score','FontWeight','bold');
grid on;
hold on;
bar(2,dbn_case1(1,10),'m')
hold on;
bar(3,cnn_case1(1,10),'g')
x = 1:1:3;
y=[sae_case1(1,10) dbn_case1(1,10) cnn_case1(1,10)];
err=[sae_case1(2,10) dbn_case1(2,10) cnn_case1(2,10)];
errorbar(x,y,err);

%accuracy
subplot(2,4,5);
bar(1,sae_case2(1,4),'b')
title('(e)','FontWeight','bold');
set(gca,'XTick',1:1:3);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'SAE','DBN','CNN'});
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
bar(2,dbn_case2(1,4),'m')
hold on;
bar(3,cnn_case2(1,4),'g')
x = 1:1:3;
y=[sae_case2(1,4) dbn_case2(1,4) cnn_case2(1,4)];
err=[sae_case2(2,4) dbn_case2(2,4) cnn_case2(2,4)];
errorbar(x,y,err);

%sensitivity
subplot(2,4,6);
bar(1,sae_case2(1,6),'b')
title('(f)','FontWeight','bold');
set(gca,'XTick',1:1:3);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'SAE','DBN','CNN'});
ylabel('Sensitivity','FontWeight','bold');
grid on;
hold on;
bar(2,dbn_case2(1,6),'m')
hold on;
bar(3,cnn_case2(1,6),'g')
x = 1:1:3;
y=[sae_case2(1,6) dbn_case2(1,6) cnn_case2(1,6)];
err=[sae_case2(2,6) dbn_case2(2,6) cnn_case2(2,6)];
errorbar(x,y,err);

%spe
subplot(2,4,7);
bar(1,sae_case2(1,7),'b')
title('(g)','FontWeight','bold');
set(gca,'XTick',1:1:3);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'SAE','DBN','CNN'});
ylabel('Specificity','FontWeight','bold');
grid on;
hold on;
bar(2,dbn_case2(1,7),'m')
hold on;
bar(3,cnn_case2(1,7),'g')
x = 1:1:3;
y=[sae_case2(1,7) dbn_case2(1,7) cnn_case2(1,7)];
err=[sae_case2(2,7) dbn_case2(2,7) cnn_case2(2,7)];
errorbar(x,y,err);

%f1
subplot(2,4,8);
bar(1,sae_case2(1,10),'b')
title('(h)','FontWeight','bold');
set(gca,'XTick',1:1:3);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'SAE','DBN','CNN'});
ylabel('F1-score','FontWeight','bold');
grid on;
hold on;
bar(2,dbn_case2(1,10),'m')
hold on;
bar(3,cnn_case2(1,10),'g')
x = 1:1:3;
y=[sae_case2(1,10) dbn_case2(1,10) cnn_case2(1,10)];
err=[sae_case2(2,10) dbn_case2(2,10) cnn_case2(2,10)];
errorbar(x,y,err);
