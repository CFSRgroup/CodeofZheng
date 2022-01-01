%分类器间做Wilcoxon检验，每种任务两个阶段的值平均，然后再做分析
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

%分类器间wilcoxon检验
% %ED-ELM  NB
% [p,h,stats]=signrank(ed_elm_acc,nb_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,nb_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,nb_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,nb_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,nb_npv,'method','approximate')
% 
% %ED-ELM  LR
% [p,h,stats]=signrank(ed_elm_acc,lr_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,lr_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,lr_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,lr_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,lr_npv,'method','approximate')
% 
% %ED-ELM  KNN
% [p,h,stats]=signrank(ed_elm_acc,knn_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,knn_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,knn_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,knn_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,knn_npv,'method','approximate')
% 
% %ED-ELM  ANN
% [p,h,stats]=signrank(ed_elm_acc,ann_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,ann_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,ann_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,ann_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,ann_npv,'method','approximate')
% 
% %ED-ELM  ELM
% [p,h,stats]=signrank(ed_elm_acc,elm_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,elm_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,elm_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,elm_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,elm_npv,'method','approximate')
% 
% %ED-ELM  LSSVM
% [p,h,stats]=signrank(ed_elm_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,lssvm_npv,'method','approximate')

% %NB  LR
% [p,h,stats]=signrank(nb_acc,lr_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,lr_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,lr_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,lr_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,lr_npv,'method','approximate')
% 
% %NB  KNN
% [p,h,stats]=signrank(nb_acc,knn_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,knn_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,knn_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,knn_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,knn_npv,'method','approximate')
% 
% %NB  ANN
% [p,h,stats]=signrank(nb_acc,ann_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,ann_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,ann_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,ann_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,ann_npv,'method','approximate')
% 
% %NB  ELM
% [p,h,stats]=signrank(nb_acc,elm_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,elm_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,elm_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,elm_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,elm_npv,'method','approximate')
% 
% %NB  LSSVM
% [p,h,stats]=signrank(nb_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,lssvm_npv,'method','approximate')

% %LR  KNN
% [p,h,stats]=signrank(lr_acc,knn_acc,'method','approximate')
% [p,h,stats]=signrank(lr_sen,knn_sen,'method','approximate')
% [p,h,stats]=signrank(lr_spe,knn_spe,'method','approximate')
% [p,h,stats]=signrank(lr_pre,knn_pre,'method','approximate')
% [p,h,stats]=signrank(lr_npv,knn_npv,'method','approximate')
% 
% %LR  ANN
% [p,h,stats]=signrank(lr_acc,ann_acc,'method','approximate')
% [p,h,stats]=signrank(lr_sen,ann_sen,'method','approximate')
% [p,h,stats]=signrank(lr_spe,ann_spe,'method','approximate')
% [p,h,stats]=signrank(lr_pre,ann_pre,'method','approximate')
% [p,h,stats]=signrank(lr_npv,ann_npv,'method','approximate')
% 
% %LR  ELM
% [p,h,stats]=signrank(lr_acc,elm_acc,'method','approximate')
% [p,h,stats]=signrank(lr_sen,elm_sen,'method','approximate')
% [p,h,stats]=signrank(lr_spe,elm_spe,'method','approximate')
% [p,h,stats]=signrank(lr_pre,elm_pre,'method','approximate')
% [p,h,stats]=signrank(lr_npv,elm_npv,'method','approximate')
% 
% %LR  LSSVM
% [p,h,stats]=signrank(lr_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(lr_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(lr_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(lr_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(lr_npv,lssvm_npv,'method','approximate')
% 
% %KNN  ANN
% [p,h,stats]=signrank(knn_acc,ann_acc,'method','approximate')
% [p,h,stats]=signrank(knn_sen,ann_sen,'method','approximate')
% [p,h,stats]=signrank(knn_spe,ann_spe,'method','approximate')
% [p,h,stats]=signrank(knn_pre,ann_pre,'method','approximate')
% [p,h,stats]=signrank(knn_npv,ann_npv,'method','approximate')
% 
% %KNN  ELM
% [p,h,stats]=signrank(knn_acc,elm_acc,'method','approximate')
% [p,h,stats]=signrank(knn_sen,elm_sen,'method','approximate')
% [p,h,stats]=signrank(knn_spe,elm_spe,'method','approximate')
% [p,h,stats]=signrank(knn_pre,elm_pre,'method','approximate')
% [p,h,stats]=signrank(knn_npv,elm_npv,'method','approximate')
% 
% %KNN  LSSVM
% [p,h,stats]=signrank(knn_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(knn_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(knn_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(knn_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(knn_npv,lssvm_npv,'method','approximate')
% 
% %ANN  ELM
% [p,h,stats]=signrank(ann_acc,elm_acc,'method','approximate')
% [p,h,stats]=signrank(ann_sen,elm_sen,'method','approximate')
% [p,h,stats]=signrank(ann_spe,elm_spe,'method','approximate')
% [p,h,stats]=signrank(ann_pre,elm_pre,'method','approximate')
% [p,h,stats]=signrank(ann_npv,elm_npv,'method','approximate')
% 
% %ANN  LSSVM
% [p,h,stats]=signrank(ann_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(ann_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(ann_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(ann_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(ann_npv,lssvm_npv,'method','approximate')
% 
% %ELM  LSSVM
% [p,h,stats]=signrank(elm_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(elm_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(elm_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(elm_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(elm_npv,lssvm_npv,'method','approximate')

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

% 分类器间wilcoxon检验
% %ED-ELM  NB
% [p,h,stats]=signrank(ed_elm_acc,nb_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,nb_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,nb_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,nb_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,nb_npv,'method','approximate')
% 
% %ED-ELM  LR
% [p,h,stats]=signrank(ed_elm_acc,lr_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,lr_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,lr_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,lr_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,lr_npv,'method','approximate')
% 
% %ED-ELM  KNN
% [p,h,stats]=signrank(ed_elm_acc,knn_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,knn_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,knn_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,knn_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,knn_npv,'method','approximate')
% 
% %ED-ELM  ANN
% [p,h,stats]=signrank(ed_elm_acc,ann_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,ann_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,ann_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,ann_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,ann_npv,'method','approximate')
% 
% %ED-ELM  ELM
% [p,h,stats]=signrank(ed_elm_acc,elm_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,elm_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,elm_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,elm_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,elm_npv,'method','approximate')
% 
% %ED-ELM  LSSVM
% [p,h,stats]=signrank(ed_elm_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(ed_elm_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(ed_elm_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(ed_elm_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(ed_elm_npv,lssvm_npv,'method','approximate')

% %NB  LR
% [p,h,stats]=signrank(nb_acc,lr_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,lr_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,lr_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,lr_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,lr_npv,'method','approximate')
% 
% %NB  KNN
% [p,h,stats]=signrank(nb_acc,knn_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,knn_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,knn_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,knn_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,knn_npv,'method','approximate')
% 
% %NB  ANN
% [p,h,stats]=signrank(nb_acc,ann_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,ann_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,ann_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,ann_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,ann_npv,'method','approximate')
% 
% %NB  ELM
% [p,h,stats]=signrank(nb_acc,elm_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,elm_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,elm_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,elm_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,elm_npv,'method','approximate')
% 
% %NB  LSSVM
% [p,h,stats]=signrank(nb_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(nb_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(nb_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(nb_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(nb_npv,lssvm_npv,'method','approximate')
% 
% %LR  KNN
% [p,h,stats]=signrank(lr_acc,knn_acc,'method','approximate')
% [p,h,stats]=signrank(lr_sen,knn_sen,'method','approximate')
% [p,h,stats]=signrank(lr_spe,knn_spe,'method','approximate')
% [p,h,stats]=signrank(lr_pre,knn_pre,'method','approximate')
% [p,h,stats]=signrank(lr_npv,knn_npv,'method','approximate')
% 
% %LR  ANN
% [p,h,stats]=signrank(lr_acc,ann_acc,'method','approximate')
% [p,h,stats]=signrank(lr_sen,ann_sen,'method','approximate')
% [p,h,stats]=signrank(lr_spe,ann_spe,'method','approximate')
% [p,h,stats]=signrank(lr_pre,ann_pre,'method','approximate')
% [p,h,stats]=signrank(lr_npv,ann_npv,'method','approximate')
% 
% %LR  ELM
% [p,h,stats]=signrank(lr_acc,elm_acc,'method','approximate')
% [p,h,stats]=signrank(lr_sen,elm_sen,'method','approximate')
% [p,h,stats]=signrank(lr_spe,elm_spe,'method','approximate')
% [p,h,stats]=signrank(lr_pre,elm_pre,'method','approximate')
% [p,h,stats]=signrank(lr_npv,elm_npv,'method','approximate')
% 
% %LR  LSSVM
% [p,h,stats]=signrank(lr_acc,lssvm_acc,'method','approximate')
% [p,h,stats]=signrank(lr_sen,lssvm_sen,'method','approximate')
% [p,h,stats]=signrank(lr_spe,lssvm_spe,'method','approximate')
% [p,h,stats]=signrank(lr_pre,lssvm_pre,'method','approximate')
% [p,h,stats]=signrank(lr_npv,lssvm_npv,'method','approximate')
% 
%KNN  ANN
[p,h,stats]=signrank(knn_acc,ann_acc,'method','approximate')
[p,h,stats]=signrank(knn_sen,ann_sen,'method','approximate')
[p,h,stats]=signrank(knn_spe,ann_spe,'method','approximate')
[p,h,stats]=signrank(knn_pre,ann_pre,'method','approximate')
[p,h,stats]=signrank(knn_npv,ann_npv,'method','approximate')

%KNN  ELM
[p,h,stats]=signrank(knn_acc,elm_acc,'method','approximate')
[p,h,stats]=signrank(knn_sen,elm_sen,'method','approximate')
[p,h,stats]=signrank(knn_spe,elm_spe,'method','approximate')
[p,h,stats]=signrank(knn_pre,elm_pre,'method','approximate')
[p,h,stats]=signrank(knn_npv,elm_npv,'method','approximate')

%KNN  LSSVM
[p,h,stats]=signrank(knn_acc,lssvm_acc,'method','approximate')
[p,h,stats]=signrank(knn_sen,lssvm_sen,'method','approximate')
[p,h,stats]=signrank(knn_spe,lssvm_spe,'method','approximate')
[p,h,stats]=signrank(knn_pre,lssvm_pre,'method','approximate')
[p,h,stats]=signrank(knn_npv,lssvm_npv,'method','approximate')

%ANN  ELM
[p,h,stats]=signrank(ann_acc,elm_acc,'method','approximate')
[p,h,stats]=signrank(ann_sen,elm_sen,'method','approximate')
[p,h,stats]=signrank(ann_spe,elm_spe,'method','approximate')
[p,h,stats]=signrank(ann_pre,elm_pre,'method','approximate')
[p,h,stats]=signrank(ann_npv,elm_npv,'method','approximate')

%ANN  LSSVM
[p,h,stats]=signrank(ann_acc,lssvm_acc,'method','approximate')
[p,h,stats]=signrank(ann_sen,lssvm_sen,'method','approximate')
[p,h,stats]=signrank(ann_spe,lssvm_spe,'method','approximate')
[p,h,stats]=signrank(ann_pre,lssvm_pre,'method','approximate')
[p,h,stats]=signrank(ann_npv,lssvm_npv,'method','approximate')

%ELM  LSSVM
[p,h,stats]=signrank(elm_acc,lssvm_acc,'method','approximate')
[p,h,stats]=signrank(elm_sen,lssvm_sen,'method','approximate')
[p,h,stats]=signrank(elm_spe,lssvm_spe,'method','approximate')
[p,h,stats]=signrank(elm_pre,lssvm_pre,'method','approximate')
[p,h,stats]=signrank(elm_npv,lssvm_npv,'method','approximate')

