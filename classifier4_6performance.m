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

%lssvm
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task1_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task1_session2
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task2_session2

lssvm_case1=(per_index_single1+per_index_single2)./2;
lssvm_case2=(per_index_single3+per_index_single4)./2;

%elm
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session1
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session2
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session2

elm_case1=(per_index_single1+per_index_single2)./2;
elm_case2=(per_index_single3+per_index_single4)./2;

%ED_ELM
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case1_result
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case2_result
ED_ELM_case1=case1_per_single_mean;
ED_ELM_case2=case2_per_single_mean;


save F:\matlab\trial_procedure\study_1\classifier4_6performance\data1 sae_case1 lssvm_case1 elm_case1 ED_ELM_case1
save F:\matlab\trial_procedure\study_1\classifier4_6performance\data2 sae_case2 lssvm_case2 elm_case2 ED_ELM_case2

% %%case1
% %accuracy
% subplot(2,3,1);
% plot(sae_case1(:,4),'ob');
% title('(a)','FontWeight','bold');
% set(gca,'XTick',1:1:8);   %图形编辑中，XLim改为0:9
% set(gca,'XTickLabel',{'A','B','C','D','E','F','G','H'});
% xlabel('Subject index','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(lssvm_case1(:,4),'^m');
% hold on;
% plot(elm_case1(:,4),'sg');
% hold on;
% plot(ED_ELM_case1(:,4),'dr');
% 
% %sensitivity
% subplot(2,3,2);
% plot(sae_case1(:,6),'ob');
% title('(b)','FontWeight','bold');
% set(gca,'XTick',1:1:8);   %图形编辑中，XLim改为0:9
% set(gca,'XTickLabel',{'A','B','C','D','E','F','G','H'});
% xlabel('Subject index','FontWeight','bold');
% ylabel('Sensitivity','FontWeight','bold');
% grid on;
% hold on;
% plot(lssvm_case1(:,6),'^m');
% hold on;
% plot(elm_case1(:,6),'sg');
% hold on;
% plot(ED_ELM_case1(:,5),'dr');
% 
% %spe
% subplot(2,3,3);
% plot(sae_case1(:,7),'ob');
% title('(c)','FontWeight','bold');
% set(gca,'XTick',1:1:8);   %图形编辑中，XLim改为0:9
% set(gca,'XTickLabel',{'A','B','C','D','E','F','G','H'});
% xlabel('Subject index','FontWeight','bold');
% ylabel('Specificity','FontWeight','bold');
% grid on;
% hold on;
% plot(lssvm_case1(:,7),'^m');
% hold on;
% plot(elm_case1(:,7),'sg');
% hold on;
% plot(ED_ELM_case1(:,6),'dr');
% 
% %pre
% subplot(2,3,4);
% plot(sae_case1(:,8),'ob');
% title('(d)','FontWeight','bold');
% set(gca,'XTick',1:1:8);   %图形编辑中，XLim改为0:9
% set(gca,'XTickLabel',{'A','B','C','D','E','F','G','H'});
% xlabel('Subject index','FontWeight','bold');
% ylabel('Precision','FontWeight','bold');
% grid on;
% hold on;
% plot(lssvm_case1(:,8),'^m');
% hold on;
% plot(elm_case1(:,8),'sg');
% hold on;
% plot(ED_ELM_case1(:,7),'dr');
% 
% %npv
% subplot(2,3,5);
% plot(sae_case1(:,9),'ob');
% title('(e)','FontWeight','bold');
% set(gca,'XTick',1:1:8);   %图形编辑中，XLim改为0:9
% set(gca,'XTickLabel',{'A','B','C','D','E','F','G','H'});
% xlabel('Subject index','FontWeight','bold');
% ylabel('NPV','FontWeight','bold');
% grid on;
% hold on;
% plot(lssvm_case1(:,9),'^m');
% hold on;
% plot(elm_case1(:,9),'sg');
% hold on;
% plot(ED_ELM_case1(:,8),'dr');
% 
% %f1
% subplot(2,3,6);
% plot(sae_case1(:,10),'ob');
% title('(f)','FontWeight','bold');
% set(gca,'XTick',1:1:8);   %图形编辑中，XLim改为0:9
% set(gca,'XTickLabel',{'A','B','C','D','E','F','G','H'});
% xlabel('Subject index','FontWeight','bold');
% ylabel('F1-score','FontWeight','bold');
% grid on;
% hold on;
% plot(lssvm_case1(:,10),'^m');
% hold on;
% plot(elm_case1(:,10),'sg');
% hold on;
% plot(ED_ELM_case1(:,9),'dr');
% 
% legend('SAE','LSSVM','ELM','ED-ELM');



%%case2
%accuracy
subplot(2,3,1);
plot(sae_case2(:,4),'ob');
title('(a)','FontWeight','bold');
set(gca,'XTick',1:1:6);   %图形编辑中，XLim改为0:7
set(gca,'XTickLabel',{'I','J','K','L','M','N'});
xlabel('Subject index','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
plot(lssvm_case2(:,4),'^m');
hold on;
plot(elm_case2(:,4),'sg');
hold on;
plot(ED_ELM_case2(:,4),'dr');

%sensitivity
subplot(2,3,2);
plot(sae_case2(:,6),'ob');
title('(b)','FontWeight','bold');
set(gca,'XTick',1:1:6);   %图形编辑中，XLim改为0:7
set(gca,'XTickLabel',{'I','J','K','L','M','N'});
xlabel('Subject index','FontWeight','bold');
ylabel('Sensitivity','FontWeight','bold');
grid on;
hold on;
plot(lssvm_case2(:,6),'^m');
hold on;
plot(elm_case2(:,6),'sg');
hold on;
plot(ED_ELM_case2(:,5),'dr');

%spe
subplot(2,3,3);
plot(sae_case2(:,7),'ob');
title('(c)','FontWeight','bold');
set(gca,'XTick',1:1:6);   %图形编辑中，XLim改为0:7
set(gca,'XTickLabel',{'I','J','K','L','M','N'});
xlabel('Subject index','FontWeight','bold');
ylabel('Specificity','FontWeight','bold');
grid on;
hold on;
plot(lssvm_case2(:,7),'^m');
hold on;
plot(elm_case2(:,7),'sg');
hold on;
plot(ED_ELM_case2(:,6),'dr');

%pre
subplot(2,3,4);
plot(sae_case2(:,8),'ob');
title('(d)','FontWeight','bold');
set(gca,'XTick',1:1:6);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'I','J','K','L','M','N'});
xlabel('Subject index','FontWeight','bold');
ylabel('Precision','FontWeight','bold');
grid on;
hold on;
plot(lssvm_case2(:,8),'^m');
hold on;
plot(elm_case2(:,8),'sg');
hold on;
plot(ED_ELM_case2(:,7),'dr');

%npv
subplot(2,3,5);
plot(sae_case2(:,9),'ob');
title('(e)','FontWeight','bold');
set(gca,'XTick',1:1:6);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'I','J','K','L','M','N'});
xlabel('Subject index','FontWeight','bold');
ylabel('NPV','FontWeight','bold');
grid on;
hold on;
plot(lssvm_case2(:,9),'^m');
hold on;
plot(elm_case2(:,9),'sg');
hold on;
plot(ED_ELM_case2(:,8),'dr');

%f1
subplot(2,3,6);
plot(sae_case2(:,10),'ob');
title('(f)','FontWeight','bold');
set(gca,'XTick',1:1:6);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'I','J','K','L','M','N'});
xlabel('Subject index','FontWeight','bold');
ylabel('F1-score','FontWeight','bold');
grid on;
hold on;
plot(lssvm_case2(:,10),'^m');
hold on;
plot(elm_case2(:,10),'sg');
hold on;
plot(ED_ELM_case2(:,9),'dr');

legend('SAE','LSSVM','ELM','ED-ELM');
