%分类器间做two_ttest检验，每种任务两个阶段的值平均，然后再做分析
clc;
clear;
close all;
warning off;

% %数据准备
% %case1
% %ed_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case1_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case1_session2_result
% ed_elm_acc=(case1_session1_per(:,4)+case1_session2_per(:,4))./2;
% ed_elm_sen=(case1_session1_per(:,5)+case1_session2_per(:,5))./2;
% ed_elm_spe=(case1_session1_per(:,6)+case1_session2_per(:,6))./2;
% ed_elm_pre=(case1_session1_per(:,7)+case1_session2_per(:,7))./2;
% ed_elm_npv=(case1_session1_per(:,8)+case1_session2_per(:,8))./2;
% 
% %nb
% load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task1_session2
% nb_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% nb_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% nb_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% nb_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% nb_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %lr
% load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task1_session2
% lr_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% lr_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% lr_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% lr_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% lr_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %knn
% load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task1_session2
% knn_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% knn_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% knn_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% knn_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% knn_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %ann
% load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task1_session2
% ann_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% ann_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% ann_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% ann_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% ann_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %elm
% load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session2
% elm_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% elm_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% elm_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% elm_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% elm_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %lssvm
% load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task1_session2
% lssvm_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% lssvm_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% lssvm_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% lssvm_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% lssvm_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %lpp_nb
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_nb_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_nb_cross_subject\task1_session2
% lpp_nb_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% lpp_nb_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% lpp_nb_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% lpp_nb_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% lpp_nb_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %lpp_lr
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task1_session2
% lpp_lr_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% lpp_lr_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% lpp_lr_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% lpp_lr_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% lpp_lr_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %lpp_knn
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_knn_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_knn_cross_subject\task1_session2
% lpp_knn_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% lpp_knn_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% lpp_knn_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% lpp_knn_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% lpp_knn_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %lpp_ann
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_ann_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_ann_cross_subject\task1_session2
% lpp_ann_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% lpp_ann_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% lpp_ann_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% lpp_ann_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% lpp_ann_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %lpp_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task1_session2
% lpp_elm_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% lpp_elm_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% lpp_elm_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% lpp_elm_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% lpp_elm_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %lpp_lssvm
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lssvm_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lssvm_cross_subject\task1_session2
% lpp_lssvm_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% lpp_lssvm_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% lpp_lssvm_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% lpp_lssvm_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% lpp_lssvm_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %h_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task1_session2
% h_elm_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% h_elm_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% h_elm_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% h_elm_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% h_elm_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %sae
% load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task1_session2
% sae_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% sae_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% sae_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% sae_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% sae_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;

% %dbn
% load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task1_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task1_session2
% dbn_acc=(per_index_single1(:,4)+per_index_single2(:,4))./2;
% dbn_sen=(per_index_single1(:,6)+per_index_single2(:,6))./2;
% dbn_spe=(per_index_single1(:,7)+per_index_single2(:,7))./2;
% dbn_pre=(per_index_single1(:,8)+per_index_single2(:,8))./2;
% dbn_npv=(per_index_single1(:,9)+per_index_single2(:,9))./2;
% 
% %b_sdae
% load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_session2_result
% b_sdae_acc=(case1_session1_per(:,4)+case1_session2_per(:,4))./2;
% b_sdae_sen=(case1_session1_per(:,5)+case1_session2_per(:,5))./2;
% b_sdae_spe=(case1_session1_per(:,6)+case1_session2_per(:,6))./2;
% b_sdae_pre=(case1_session1_per(:,7)+case1_session2_per(:,7))./2;
% b_sdae_npv=(case1_session1_per(:,8)+case1_session2_per(:,8))./2;
% 
% %b_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_session2_result
% b_elm_acc=(case1_session1_per(:,4)+case1_session2_per(:,4))./2;
% b_elm_sen=(case1_session1_per(:,5)+case1_session2_per(:,5))./2;
% b_elm_spe=(case1_session1_per(:,6)+case1_session2_per(:,6))./2;
% b_elm_pre=(case1_session1_per(:,7)+case1_session2_per(:,7))./2;
% b_elm_npv=(case1_session1_per(:,8)+case1_session2_per(:,8))./2;

% 分类器间two_ttest检验
% %ED-ELM  NB
% [h,p,ci,stats]=ttest(ed_elm_acc,nb_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,nb_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,nb_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,nb_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,nb_npv)
% 
% %ED-ELM  LR
% [h,p,ci,stats]=ttest(ed_elm_acc,lr_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lr_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lr_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lr_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lr_npv)
% 
% %ED-ELM  KNN
% [h,p,ci,stats]=ttest(ed_elm_acc,knn_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,knn_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,knn_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,knn_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,knn_npv)
% 
% %ED-ELM  ANN
% [h,p,ci,stats]=ttest(ed_elm_acc,ann_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,ann_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,ann_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,ann_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,ann_npv)
% 
% %ED-ELM  ELM
% [h,p,ci,stats]=ttest(ed_elm_acc,elm_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,elm_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,elm_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,elm_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,elm_npv)
% 
% %ED-ELM  LSSVM
% [h,p,ci,stats]=ttest(ed_elm_acc,lssvm_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lssvm_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lssvm_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lssvm_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lssvm_npv)

% %ED-ELM  lpp_NB
% [h,p,ci,stats]=ttest(ed_elm_acc,lpp_nb_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lpp_nb_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lpp_nb_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lpp_nb_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lpp_nb_npv)
% 
% %ED-ELM  lpp_LR
% [h,p,ci,stats]=ttest(ed_elm_acc,lpp_lr_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lpp_lr_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lpp_lr_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lpp_lr_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lpp_lr_npv)
% 
% %ED-ELM  lpp_KNN
% [h,p,ci,stats]=ttest(ed_elm_acc,lpp_knn_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lpp_knn_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lpp_knn_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lpp_knn_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lpp_knn_npv)
% 
% %ED-ELM  lpp_ANN
% [h,p,ci,stats]=ttest(ed_elm_acc,lpp_ann_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lpp_ann_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lpp_ann_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lpp_ann_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lpp_ann_npv)
% 
% %ED-ELM  lpp_ELM
% [h,p,ci,stats]=ttest(ed_elm_acc,lpp_elm_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lpp_elm_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lpp_elm_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lpp_elm_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lpp_elm_npv)
% 
% %ED-ELM  lpp_LSSVM
% [h,p,ci,stats]=ttest(ed_elm_acc,lpp_lssvm_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lpp_lssvm_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lpp_lssvm_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lpp_lssvm_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lpp_lssvm_npv)
% 
% %ED-ELM  h_elm
% [h,p,ci,stats]=ttest(ed_elm_acc,h_elm_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,h_elm_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,h_elm_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,h_elm_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,h_elm_npv)
% 
% %ED-ELM  sae
% [h,p,ci,stats]=ttest(ed_elm_acc,sae_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,sae_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,sae_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,sae_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,sae_npv)

% %ED-ELM  dbn
% [p,h,stats]=signrank(ed_elm_acc,dbn_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,dbn_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,dbn_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,dbn_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,dbn_npv,'method','approximate')
% 
% %ED-ELM  b_sdae
% [p,h,stats]=signrank(ed_elm_acc,b_sdae_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,b_sdae_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,b_sdae_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,b_sdae_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,b_sdae_npv,'method','approximate')
% 
% %ED-ELM  b_elm
% [p,h,stats]=signrank(ed_elm_acc,b_elm_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,b_elm_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,b_elm_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,b_elm_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,b_elm_npv,'method','approximate')



%数据准备
%case2
%ed_elm
load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case2_session1_result
load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case2_session2_result
ed_elm_acc=(case2_session1_per(:,4)+case2_session2_per(:,4))./2;
ed_elm_sen=(case2_session1_per(:,5)+case2_session2_per(:,5))./2;
ed_elm_spe=(case2_session1_per(:,6)+case2_session2_per(:,6))./2;
ed_elm_pre=(case2_session1_per(:,7)+case2_session2_per(:,7))./2;
ed_elm_npv=(case2_session1_per(:,8)+case2_session2_per(:,8))./2;

%nb
load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task2_session2
nb_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
nb_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
nb_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
nb_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
nb_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%lr
load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task2_session2
lr_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
lr_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
lr_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
lr_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
lr_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%knn
load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task2_session2
knn_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
knn_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
knn_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
knn_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
knn_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%ann
load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task2_session2
ann_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
ann_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
ann_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
ann_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
ann_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%elm
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session2
elm_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
elm_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
elm_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
elm_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
elm_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%lssvm
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task2_session2
lssvm_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
lssvm_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
lssvm_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
lssvm_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
lssvm_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%lpp_nb
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_nb_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_nb_cross_subject\task2_session2
lpp_nb_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
lpp_nb_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
lpp_nb_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
lpp_nb_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
lpp_nb_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%lpp_lr
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task2_session2
lpp_lr_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
lpp_lr_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
lpp_lr_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
lpp_lr_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
lpp_lr_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%lpp_knn
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_knn_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_knn_cross_subject\task2_session2
lpp_knn_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
lpp_knn_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
lpp_knn_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
lpp_knn_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
lpp_knn_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%lpp_ann
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_ann_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_ann_cross_subject\task2_session2
lpp_ann_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
lpp_ann_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
lpp_ann_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
lpp_ann_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
lpp_ann_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%lpp_elm
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task2_session2
lpp_elm_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
lpp_elm_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
lpp_elm_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
lpp_elm_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
lpp_elm_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%lpp_lssvm
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lssvm_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lssvm_cross_subject\task2_session2
lpp_lssvm_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
lpp_lssvm_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
lpp_lssvm_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
lpp_lssvm_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
lpp_lssvm_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%h_elm
load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task2_session2
h_elm_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
h_elm_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
h_elm_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
h_elm_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
h_elm_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

%sae
load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task2_session1
load F:\matlab\trial_procedure\study_1\data_analysis\sae_cross_subject\task2_session2
sae_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
sae_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
sae_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
sae_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
sae_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;

% %dbn
% load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task2_session1
% load F:\matlab\trial_procedure\study_1\data_analysis\dbn_cross_subject\task2_session2
% dbn_acc=(per_index_single3(:,4)+per_index_single4(:,4))./2;
% dbn_sen=(per_index_single3(:,6)+per_index_single4(:,6))./2;
% dbn_spe=(per_index_single3(:,7)+per_index_single4(:,7))./2;
% dbn_pre=(per_index_single3(:,8)+per_index_single4(:,8))./2;
% dbn_npv=(per_index_single3(:,9)+per_index_single4(:,9))./2;
% 
% %b_sdae
% load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_session2_result
% b_sdae_acc=(case2_session1_per(:,4)+case2_session2_per(:,4))./2;
% b_sdae_sen=(case2_session1_per(:,5)+case2_session2_per(:,5))./2;
% b_sdae_spe=(case2_session1_per(:,6)+case2_session2_per(:,6))./2;
% b_sdae_pre=(case2_session1_per(:,7)+case2_session2_per(:,7))./2;
% b_sdae_npv=(case2_session1_per(:,8)+case2_session2_per(:,8))./2;
% 
% %b_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_session2_result
% b_elm_acc=(case2_session1_per(:,4)+case2_session2_per(:,4))./2;
% b_elm_sen=(case2_session1_per(:,5)+case2_session2_per(:,5))./2;
% b_elm_spe=(case2_session1_per(:,6)+case2_session2_per(:,6))./2;
% b_elm_pre=(case2_session1_per(:,7)+case2_session2_per(:,7))./2;
% b_elm_npv=(case2_session1_per(:,8)+case2_session2_per(:,8))./2;

% 分类器间two_ttest检验
% % ED-ELM  NB
% [h,p,ci,stats]=ttest(ed_elm_acc,nb_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,nb_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,nb_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,nb_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,nb_npv)
% 
% %ED-ELM  LR
% [h,p,ci,stats]=ttest(ed_elm_acc,lr_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lr_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lr_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lr_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lr_npv)
% 
% %ED-ELM  KNN
% [h,p,ci,stats]=ttest(ed_elm_acc,knn_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,knn_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,knn_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,knn_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,knn_npv)
% 
% %ED-ELM  ANN
% [h,p,ci,stats]=ttest(ed_elm_acc,ann_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,ann_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,ann_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,ann_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,ann_npv)
% 
% %ED-ELM  ELM
% [h,p,ci,stats]=ttest(ed_elm_acc,elm_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,elm_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,elm_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,elm_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,elm_npv)
% 
% %ED-ELM  LSSVM
% [h,p,ci,stats]=ttest(ed_elm_acc,lssvm_acc)
% [h,p,ci,stats]=ttest(ed_elm_sen,lssvm_sen)
% [h,p,ci,stats]=ttest(ed_elm_spe,lssvm_spe)
% [h,p,ci,stats]=ttest(ed_elm_pre,lssvm_pre)
% [h,p,ci,stats]=ttest(ed_elm_npv,lssvm_npv)

%ED-ELM  lpp_NB
[h,p,ci,stats]=ttest(ed_elm_acc,lpp_nb_acc)
[h,p,ci,stats]=ttest(ed_elm_sen,lpp_nb_sen)
[h,p,ci,stats]=ttest(ed_elm_spe,lpp_nb_spe)
[h,p,ci,stats]=ttest(ed_elm_pre,lpp_nb_pre)
[h,p,ci,stats]=ttest(ed_elm_npv,lpp_nb_npv)

%ED-ELM  lpp_LR
[h,p,ci,stats]=ttest(ed_elm_acc,lpp_lr_acc)
[h,p,ci,stats]=ttest(ed_elm_sen,lpp_lr_sen)
[h,p,ci,stats]=ttest(ed_elm_spe,lpp_lr_spe)
[h,p,ci,stats]=ttest(ed_elm_pre,lpp_lr_pre)
[h,p,ci,stats]=ttest(ed_elm_npv,lpp_lr_npv)

%ED-ELM  lpp_KNN
[h,p,ci,stats]=ttest(ed_elm_acc,lpp_knn_acc)
[h,p,ci,stats]=ttest(ed_elm_sen,lpp_knn_sen)
[h,p,ci,stats]=ttest(ed_elm_spe,lpp_knn_spe)
[h,p,ci,stats]=ttest(ed_elm_pre,lpp_knn_pre)
[h,p,ci,stats]=ttest(ed_elm_npv,lpp_knn_npv)

%ED-ELM  lpp_ANN
[h,p,ci,stats]=ttest(ed_elm_acc,lpp_ann_acc)
[h,p,ci,stats]=ttest(ed_elm_sen,lpp_ann_sen)
[h,p,ci,stats]=ttest(ed_elm_spe,lpp_ann_spe)
[h,p,ci,stats]=ttest(ed_elm_pre,lpp_ann_pre)
[h,p,ci,stats]=ttest(ed_elm_npv,lpp_ann_npv)

%ED-ELM  lpp_ELM
[h,p,ci,stats]=ttest(ed_elm_acc,lpp_elm_acc)
[h,p,ci,stats]=ttest(ed_elm_sen,lpp_elm_sen)
[h,p,ci,stats]=ttest(ed_elm_spe,lpp_elm_spe)
[h,p,ci,stats]=ttest(ed_elm_pre,lpp_elm_pre)
[h,p,ci,stats]=ttest(ed_elm_npv,lpp_elm_npv)

%ED-ELM  lpp_LSSVM
[h,p,ci,stats]=ttest(ed_elm_acc,lpp_lssvm_acc)
[h,p,ci,stats]=ttest(ed_elm_sen,lpp_lssvm_sen)
[h,p,ci,stats]=ttest(ed_elm_spe,lpp_lssvm_spe)
[h,p,ci,stats]=ttest(ed_elm_pre,lpp_lssvm_pre)
[h,p,ci,stats]=ttest(ed_elm_npv,lpp_lssvm_npv)

%ED-ELM  h_elm
[h,p,ci,stats]=ttest(ed_elm_acc,h_elm_acc)
[h,p,ci,stats]=ttest(ed_elm_sen,h_elm_sen)
[h,p,ci,stats]=ttest(ed_elm_spe,h_elm_spe)
[h,p,ci,stats]=ttest(ed_elm_pre,h_elm_pre)
[h,p,ci,stats]=ttest(ed_elm_npv,h_elm_npv)

%ED-ELM  sae
[h,p,ci,stats]=ttest(ed_elm_acc,sae_acc)
[h,p,ci,stats]=ttest(ed_elm_sen,sae_sen)
[h,p,ci,stats]=ttest(ed_elm_spe,sae_spe)
[h,p,ci,stats]=ttest(ed_elm_pre,sae_pre)
[h,p,ci,stats]=ttest(ed_elm_npv,sae_npv)

% %ED-ELM  dbn
% [p,h,stats]=signrank(ed_elm_acc,dbn_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,dbn_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,dbn_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,dbn_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,dbn_npv,'method','approximate')
% 
% %ED-ELM  b_sdae
% [p,h,stats]=signrank(ed_elm_acc,b_sdae_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,b_sdae_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,b_sdae_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,b_sdae_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,b_sdae_npv,'method','approximate')
% 
% %ED-ELM  b_elm
% [p,h,stats]=signrank(ed_elm_acc,b_elm_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,b_elm_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,b_elm_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,b_elm_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,b_elm_npv,'method','approximate')


