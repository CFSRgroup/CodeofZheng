%统计学检验，8个分类器的精度与随机分类分类精度0.33做检验
clc;
clear;
close all;
warning off;

%case1_session1
y1=[0.33;0.33;0.33;0.33;0.33;0.33;0.33;0.33];

% %nb
% load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task1_session1
% nb1_1=per_index_single1(:,4);
% [h,p,ci,stats]=ttest(nb1_1,y1)
% 
% %lr
% load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task1_session1
% lr1_1=per_index_single1(:,4);
% [h,p,ci,stats]=ttest(lr1_1,y1)
% 
% %knn
% load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task1_session1
% knn1_1=per_index_single1(:,4);
% [h,p,ci,stats]=ttest(knn1_1,y1)
% 
% %ann
% load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task1_session1
% ann1_1=per_index_single1(:,4);
% [h,p,ci,stats]=ttest(ann1_1,y1)
% 
% %elm
% load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session1
% elm1_1=per_index_single1(:,4);
% [h,p,ci,stats]=ttest(elm1_1,y1)
% 
% %lssvm
% load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task1_session1
% lssvm1_1=per_index_single1(:,4);
% [h,p,ci,stats]=ttest(lssvm1_1,y1)
% 
% %h_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task1_session1
% h_elm1_1=per_index_single1(:,4);
% [h,p,ci,stats]=ttest(h_elm1_1,y1)
% 
% %ed_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case1_session1_result
% ed_elm1_1=case1_session1_per(:,4);
% [h,p,ci,stats]=ttest(ed_elm1_1,y1)

%case1_session2
%nb
load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task1_session2
nb1_1=per_index_single2(:,4);
[h,p,ci,stats]=ttest(nb1_1,y1)

%lr
load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task1_session2
lr1_1=per_index_single2(:,4);
[h,p,ci,stats]=ttest(lr1_1,y1)

%knn
load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task1_session2
knn1_1=per_index_single2(:,4);
[h,p,ci,stats]=ttest(knn1_1,y1)

%ann
load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task1_session2
ann1_1=per_index_single2(:,4);
[h,p,ci,stats]=ttest(ann1_1,y1)

%elm
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session2
elm1_1=per_index_single2(:,4);
[h,p,ci,stats]=ttest(elm1_1,y1)

%lssvm
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task1_session2
lssvm1_1=per_index_single2(:,4);
[h,p,ci,stats]=ttest(lssvm1_1,y1)

%h_elm
load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task1_session2
h_elm1_1=per_index_single2(:,4);
[h,p,ci,stats]=ttest(h_elm1_1,y1)

%ed_elm
load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case1_session2_result
ed_elm1_1=case1_session2_per(:,4);
[h,p,ci,stats]=ttest(ed_elm1_1,y1)


%case2_session1
y2=[0.33;0.33;0.33;0.33;0.33;0.33];

% %nb
% load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task2_session1
% nb1_1=per_index_single3(:,4);
% [h,p,ci,stats]=ttest(nb1_1,y2)
% 
% %lr
% load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task2_session1
% lr1_1=per_index_single3(:,4);
% [h,p,ci,stats]=ttest(lr1_1,y2)
% 
% %knn
% load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task2_session1
% knn1_1=per_index_single3(:,4);
% [h,p,ci,stats]=ttest(knn1_1,y2)
% 
% %ann
% load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task2_session1
% ann1_1=per_index_single3(:,4);
% [h,p,ci,stats]=ttest(ann1_1,y2)
% 
% %elm
% load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session1
% elm1_1=per_index_single3(:,4);
% [h,p,ci,stats]=ttest(elm1_1,y2)
% 
% %lssvm
% load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task2_session1
% lssvm1_1=per_index_single3(:,4);
% [h,p,ci,stats]=ttest(lssvm1_1,y2)
% 
% %h_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task2_session1
% h_elm1_1=per_index_single3(:,4);
% [h,p,ci,stats]=ttest(h_elm1_1,y2)
% 
% %ed_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case2_session1_result
% ed_elm1_1=case2_session1_per(:,4);
% [h,p,ci,stats]=ttest(ed_elm1_1,y2)



% %case2_session2
% %nb
% load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task2_session2
% nb1_1=per_index_single4(:,4);
% [h,p,ci,stats]=ttest(nb1_1,y2)
% 
% %lr
% load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task2_session2
% lr1_1=per_index_single4(:,4);
% [h,p,ci,stats]=ttest(lr1_1,y2)
% 
% %knn
% load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task2_session2
% knn1_1=per_index_single4(:,4);
% [h,p,ci,stats]=ttest(knn1_1,y2)
% 
% %ann
% load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task2_session2
% ann1_1=per_index_single4(:,4);
% [h,p,ci,stats]=ttest(ann1_1,y2)
% 
% %elm
% load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session2
% elm1_1=per_index_single4(:,4);
% [h,p,ci,stats]=ttest(elm1_1,y2)
% 
% %lssvm
% load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task2_session2
% lssvm1_1=per_index_single4(:,4);
% [h,p,ci,stats]=ttest(lssvm1_1,y2)
% 
% %h_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task2_session2
% h_elm1_1=per_index_single4(:,4);
% [h,p,ci,stats]=ttest(h_elm1_1,y2)
% 
% %ed_elm
% load F:\matlab\trial_procedure\study_1\data_analysis\ed_elm_cross_subject\case2_session2_result
% ed_elm1_1=case2_session2_per(:,4);
% [h,p,ci,stats]=ttest(ed_elm1_1,y2)



