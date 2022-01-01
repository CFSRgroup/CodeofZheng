clc;
clear;
close all;
warning off;

% load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\ann_ave_case_1.mat')
% load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\knn_ave_case_1.mat')
% load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\lr_ave_case_1.mat')
% load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\nb_ave_case_1.mat')
% 
% acc=[ann_ave_case_1(4,1) knn_ave_case_1(4,1) lr_ave_case_1(4,1) nb_ave_case_1(4,1)];
% sd=[ann_ave_case_1(4,2) knn_ave_case_1(4,2) lr_ave_case_1(4,2) nb_ave_case_1(4,2)];
% 
% bar(acc,'DisplayName','acc');
% title('(a)','FontWeight','bold');
% grid on;
% hold on;
% set(gca,'XTickLabel',{'ANN','KNN','LR','NB'},'FontWeight','bold');
% xtickangle(50);
% errorbar(acc,sd); 
% ylabel('Accuracy','FontWeight','bold');


% load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\ann_ave_case_1.mat')
% load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\knn_ave_case_1.mat')
% load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\lr_ave_case_1.mat')
% load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\nb_ave_case_1.mat')

% acc=[ann_ave_case_1(6,1) knn_ave_case_1(6,1) lr_ave_case_1(6,1) nb_ave_case_1(6,1)];
% sd=[ann_ave_case_1(6,2) knn_ave_case_1(6,2) lr_ave_case_1(6,2) nb_ave_case_1(6,2)];
% 
% bar(acc,'DisplayName','acc');
% title('(b)','FontWeight','bold');
% grid on;
% hold on;
% set(gca,'XTickLabel',{'ANN','KNN','LR','NB'},'FontWeight','bold');
% xtickangle(50);
% errorbar(acc,sd); 
% ylabel('Sensitivity','FontWeight','bold');

load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\ann_ave_case_2.mat')
load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\knn_ave_case_2.mat')
load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\lr_ave_case_2.mat')
load('F:\matlab\trial_procedure\study_1\data_analysis\error_figure\nb_ave_case_2.mat')


acc=[ann_ave_case_2(10,1) knn_ave_case_2(10,1) lr_ave_case_2(10,1) nb_ave_case_2(10,1)];
sd=[ann_ave_case_2(10,2) knn_ave_case_2(10,2) lr_ave_case_2(10,2) nb_ave_case_2(10,2)];

bar(acc,'DisplayName','acc');
title('(f)','FontWeight','bold');
grid on;
hold on;
set(gca,'XTickLabel',{'ANN','KNN','LR','NB'},'FontWeight','bold');
xtickangle(50);
errorbar(acc,sd); 
ylabel('F1-score','FontWeight','bold');