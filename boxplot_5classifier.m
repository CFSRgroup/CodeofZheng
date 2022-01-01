%箱线图绘制，5个分类器，就acc sen spe pre npv f1展开分析

clc;
clear;
close all;
warning off;

% %case1
% %sae
% load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task1_session2
% 
% sae_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% sae_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% sae_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% sae_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% sae_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% sae_f1=(per_index_single1(:,10)+per_index_single2(:,10))./2;
% 
% %dbn
% load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task1_session2
% 
% dbn_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% dbn_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% dbn_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% dbn_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% dbn_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% dbn_f1=(per_index_single1(:,10)+per_index_single2(:,10))./2;
% 
% %b-sdae
% load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_session2_result
% 
% b_sdae_acc=(case1_session1_per(:,4)+case1_session2_per(:,4))./2;
% b_sdae_sen=(case1_session1_per(:,5)+case1_session2_per(:,5))./2;
% b_sdae_spe=(case1_session1_per(:,6)+case1_session2_per(:,6))./2;
% b_sdae_pre=(case1_session1_per(:,7)+case1_session2_per(:,7))./2;
% b_sdae_npv=(case1_session1_per(:,8)+case1_session2_per(:,8))./2;
% b_sdae_f1=(case1_session1_per(:,9)+case1_session2_per(:,9))./2;
% 
% %b-elm
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_session2_result
% 
% b_elm_acc=(case1_session1_per(:,4)+case1_session2_per(:,4))./2;
% b_elm_sen=(case1_session1_per(:,5)+case1_session2_per(:,5))./2;
% b_elm_spe=(case1_session1_per(:,6)+case1_session2_per(:,6))./2;
% b_elm_pre=(case1_session1_per(:,7)+case1_session2_per(:,7))./2;
% b_elm_npv=(case1_session1_per(:,8)+case1_session2_per(:,8))./2;
% b_elm_f1=(case1_session1_per(:,9)+case1_session2_per(:,9))./2;
% 
% %ed_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case1_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case1_session2_result
% 
% ed_elm_acc=(case1_session1_per(:,4)+case1_session2_per(:,4))./2;
% ed_elm_sen=(case1_session1_per(:,5)+case1_session2_per(:,5))./2;
% ed_elm_spe=(case1_session1_per(:,6)+case1_session2_per(:,6))./2;
% ed_elm_pre=(case1_session1_per(:,7)+case1_session2_per(:,7))./2;
% ed_elm_npv=(case1_session1_per(:,8)+case1_session2_per(:,8))./2;
% ed_elm_f1=(case1_session1_per(:,9)+case1_session2_per(:,9))./2;
% 
% case1_acc=[sae_acc dbn_acc b_sdae_acc b_elm_acc ed_elm_acc];
% case1_sen=[sae_sen dbn_sen b_sdae_sen b_elm_sen ed_elm_sen];
% case1_spe=[sae_spe dbn_spe b_sdae_spe b_elm_spe ed_elm_spe];
% case1_pre=[sae_pre dbn_pre b_sdae_pre b_elm_pre ed_elm_pre];
% case1_npv=[sae_npv dbn_npv b_sdae_npv b_elm_npv ed_elm_npv];
% case1_f1=[sae_f1 dbn_f1 b_sdae_f1 b_elm_f1 ed_elm_f1];
% 
% subplot(2,3,1);
% boxplot(case1_acc);
% set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
% title('(a)','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% hold on;
% 
% subplot(2,3,2);
% boxplot(case1_sen);
% set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
% title('(b)','FontWeight','bold');
% ylabel('Sensitivity','FontWeight','bold');
% hold on;
% 
% subplot(2,3,3);
% boxplot(case1_spe);
% set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
% title('(c)','FontWeight','bold');
% ylabel('Specificity','FontWeight','bold');
% hold on;
% 
% subplot(2,3,4);
% boxplot(case1_pre);
% set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
% title('(d)','FontWeight','bold');
% ylabel('Precision','FontWeight','bold');
% hold on;
% 
% subplot(2,3,5);
% boxplot(case1_npv);
% set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
% title('(e)','FontWeight','bold');
% ylabel('NPV','FontWeight','bold');
% hold on;
% 
% subplot(2,3,6);
% boxplot(case1_f1);
% set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
% title('(f)','FontWeight','bold');
% ylabel('F1-score','FontWeight','bold');
% hold on;


%case2
%sae
load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task2_session2

sae_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
sae_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
sae_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
sae_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
sae_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;
sae_f1=(per_index_single3(:,10)+per_index_single4(:,10))./2;

%dbn
load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task2_session2

dbn_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
dbn_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
dbn_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
dbn_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
dbn_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;
dbn_f1=(per_index_single3(:,10)+per_index_single4(:,10))./2;

%b-sdae
load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_session1_result
load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_session2_result

b_sdae_acc=(case2_session1_per(:,4)+case2_session2_per(:,4))./2;
b_sdae_sen=(case2_session1_per(:,5)+case2_session2_per(:,5))./2;
b_sdae_spe=(case2_session1_per(:,6)+case2_session2_per(:,6))./2;
b_sdae_pre=(case2_session1_per(:,7)+case2_session2_per(:,7))./2;
b_sdae_npv=(case2_session1_per(:,8)+case2_session2_per(:,8))./2;
b_sdae_f1=(case2_session1_per(:,9)+case2_session2_per(:,9))./2;

%b-elm
load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_session1_result
load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_session2_result

b_elm_acc=(case2_session1_per(:,4)+case2_session2_per(:,4))./2;
b_elm_sen=(case2_session1_per(:,5)+case2_session2_per(:,5))./2;
b_elm_spe=(case2_session1_per(:,6)+case2_session2_per(:,6))./2;
b_elm_pre=(case2_session1_per(:,7)+case2_session2_per(:,7))./2;
b_elm_npv=(case2_session1_per(:,8)+case2_session2_per(:,8))./2;
b_elm_f1=(case2_session1_per(:,9)+case2_session2_per(:,9))./2;

%ed_elm
load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case2_session1_result
load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case2_session2_result

ed_elm_acc=(case2_session1_per(:,4)+case2_session2_per(:,4))./2;
ed_elm_sen=(case2_session1_per(:,5)+case2_session2_per(:,5))./2;
ed_elm_spe=(case2_session1_per(:,6)+case2_session2_per(:,6))./2;
ed_elm_pre=(case2_session1_per(:,7)+case2_session2_per(:,7))./2;
ed_elm_npv=(case2_session1_per(:,8)+case2_session2_per(:,8))./2;
ed_elm_f1=(case2_session1_per(:,9)+case2_session2_per(:,9))./2;

case2_acc=[sae_acc dbn_acc b_sdae_acc b_elm_acc ed_elm_acc];
case2_sen=[sae_sen dbn_sen b_sdae_sen b_elm_sen ed_elm_sen];
case2_spe=[sae_spe dbn_spe b_sdae_spe b_elm_spe ed_elm_spe];
case2_pre=[sae_pre dbn_pre b_sdae_pre b_elm_pre ed_elm_pre];
case2_npv=[sae_npv dbn_npv b_sdae_npv b_elm_npv ed_elm_npv];
case2_f1=[sae_f1 dbn_f1 b_sdae_f1 b_elm_f1 ed_elm_f1];

subplot(2,3,1);
boxplot(case2_acc);
set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
title('(g)','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
hold on;

subplot(2,3,2);
boxplot(case2_sen);
set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
title('(h)','FontWeight','bold');
ylabel('Sensitivity','FontWeight','bold');
hold on;

subplot(2,3,3);
boxplot(case2_spe);
set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
title('(i)','FontWeight','bold');
ylabel('Specificity','FontWeight','bold');
hold on;

subplot(2,3,4);
boxplot(case2_pre);
set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
title('(j)','FontWeight','bold');
ylabel('Precision','FontWeight','bold');
hold on;

subplot(2,3,5);
boxplot(case2_npv);
set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
title('(k)','FontWeight','bold');
ylabel('NPV','FontWeight','bold');
hold on;

subplot(2,3,6);
boxplot(case2_f1);
set(gca,'XTickLabel',{'SAE','DBN','B-SDAE','B-ELM','ED-ELM'});
title('(l)','FontWeight','bold');
ylabel('F1-score','FontWeight','bold');
hold on;