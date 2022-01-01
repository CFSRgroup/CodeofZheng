clc;
clear;
close all;
warning off;

% %(a)(b)ELM
% %case1_session1
% load F:\matlab\trial_procedure\study_1\performance_comparison\elm_cross_subject\task1_session1
% subplot(2,1,1);
% plot(plot_TrainingAccuracy,'s-g');
% title('(a) ELM: Case 1','FontWeight','bold');
% set(gca,'XTick',0:10:50);
% set(gca,'XTickLabel',{'0','100','200','300','400','500'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Number of hidden neurons','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy,'o-b');
% hold on;
% 
% %case1_session2
% load F:\matlab\trial_procedure\study_1\performance_comparison\elm_cross_subject\task1_session2
% plot(plot_TrainingAccuracy,'*-r');
% hold on;
% plot(plot_TestingAccuracy,'+-y');
% 
% legend('Training: session 1','Testing: session 1','Training: session 2','Testing: session 2');
% 
% %case2_session1
% load F:\matlab\trial_procedure\study_1\performance_comparison\elm_cross_subject\task2_session1
% subplot(2,1,2);
% plot(plot_TrainingAccuracy_1,'s-g');
% title('(b) ELM: Case 2','FontWeight','bold');
% set(gca,'XTick',0:10:50);
% set(gca,'XTickLabel',{'0','100','200','300','400','500'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Number of hidden neurons','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy_1,'o-b');
% hold on;
% 
% %case2_session2
% load F:\matlab\trial_procedure\study_1\performance_comparison\elm_cross_subject\task2_session2
% plot(plot_TrainingAccuracy_1,'*-r');
% hold on;
% plot(plot_TestingAccuracy_1,'+-y');


% %(c)(d)LSSVM
% %case1_session1
% load F:\matlab\trial_procedure\study_1\performance_comparison\lssvm_ovo_lin_cross_subject\task1_session1
% subplot(2,1,1);
% plot(plot_TrainingAccuracy,'s-g');
% title('(c) LSSVM: Case 1','FontWeight','bold');
% set(gca,'XTick',0:5:50);
% set(gca,'XTickLabel',{'2^-^2^5','2^-^2^0','2^-^1^5','2^-^1^0','2^-^5','1','2^5','2^1^0','2^1^5','2^2^0','2^2^5'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Regularization parameter','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy,'o-b');
% hold on;
% 
% %case1_session2
% load F:\matlab\trial_procedure\study_1\performance_comparison\lssvm_ovo_lin_cross_subject\task1_session2
% plot(plot_TrainingAccuracy,'*-r');
% hold on;
% plot(plot_TestingAccuracy,'+-y');
% 
% %case2_session1
% load F:\matlab\trial_procedure\study_1\performance_comparison\lssvm_ovo_lin_cross_subject\task2_session1
% subplot(2,1,2);
% plot(plot_TrainingAccuracy_1,'s-g');
% title('(d) LSSVM: Case 2','FontWeight','bold');
% set(gca,'XTick',0:5:50);
% set(gca,'XTickLabel',{'2^-^2^5','2^-^2^0','2^-^1^5','2^-^1^0','2^-^5','1','2^5','2^1^0','2^1^5','2^2^0','2^2^5'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Regularization parameter','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy_1,'o-b');
% hold on;
% 
% %case2_session2
% load F:\matlab\trial_procedure\study_1\performance_comparison\lssvm_ovo_lin_cross_subject\task2_session2
% plot(plot_TrainingAccuracy_1,'*-r');
% hold on;
% plot(plot_TestingAccuracy_1,'+-y');


% %(e)(f)KNN
% %case1_session1
% load F:\matlab\trial_procedure\study_1\performance_comparison\knn_cross_subject\task1_session1
% subplot(2,1,1);
% plot(plot_TrainingAccuracy,'s-g');
% title('(e) KNN: Case 1','FontWeight','bold');
% set(gca,'XTick',0:10:50);
% set(gca,'XTickLabel',{'0','10','20','30','40','50'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Parameter k','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy,'o-b');
% hold on;
% 
% %case1_session2
% load F:\matlab\trial_procedure\study_1\performance_comparison\knn_cross_subject\task1_session2
% plot(plot_TrainingAccuracy,'*-r');
% hold on;
% plot(plot_TestingAccuracy,'+-y');
% 
% %case2_session1
% load F:\matlab\trial_procedure\study_1\performance_comparison\knn_cross_subject\task2_session1
% subplot(2,1,2);
% plot(plot_TrainingAccuracy_1,'s-g');
% title('(f) KNN: Case 2','FontWeight','bold');
% set(gca,'XTick',0:10:50);
% set(gca,'XTickLabel',{'0','10','20','30','40','50'});
% set(gca,'YTick',0:0.05:1);
% xlabel('Parameter k','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% grid on;
% hold on;
% plot(plot_TestingAccuracy_1,'o-b');
% hold on;
% 
% %case2_session2
% load F:\matlab\trial_procedure\study_1\performance_comparison\knn_cross_subject\task2_session2
% plot(plot_TrainingAccuracy_1,'*-r');
% hold on;
% plot(plot_TestingAccuracy_1,'+-y');



%(g)(h)ANN
%case1_session1
load F:\matlab\trial_procedure\study_1\performance_comparison\ann_cross_subject\task1_session1
subplot(2,1,1);
plot(plot_TrainingAccuracy,'s-g');
title('(g) ANN: Case 1','FontWeight','bold');
set(gca,'XTick',0:10:50);
set(gca,'XTickLabel',{'0','100','200','300','400','500'});
set(gca,'YTick',0:0.05:1);
xlabel('Number of hidden neurons','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
plot(plot_TestingAccuracy,'o-b');
hold on;

%case1_session2
load F:\matlab\trial_procedure\study_1\performance_comparison\ann_cross_subject\task1_session2
plot(plot_TrainingAccuracy,'*-r');
hold on;
plot(plot_TestingAccuracy,'+-y');

%case2_session1
load F:\matlab\trial_procedure\study_1\performance_comparison\ann_cross_subject\task2_session1
subplot(2,1,2);
plot(plot_TrainingAccuracy_1,'s-g');
title('(h) ANN: Case 2','FontWeight','bold');
set(gca,'XTick',0:10:50);
set(gca,'XTickLabel',{'0','100','200','300','400','500'});
set(gca,'YTick',0:0.05:1);
xlabel('Number of hidden neurons','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
plot(plot_TestingAccuracy_1,'o-b');
hold on;

%case2_session2
load F:\matlab\trial_procedure\study_1\performance_comparison\ann_cross_subject\task2_session2
plot(plot_TrainingAccuracy_1,'*-r');
hold on;
plot(plot_TestingAccuracy_1,'+-y');






