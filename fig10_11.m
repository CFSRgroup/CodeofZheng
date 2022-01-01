clc;
clear;
close all;
warning off;

%case1
%lssvm
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));

load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

lssvm_acc=(accuracy_mean+accuracy_mean_1)./2;
lssvm_acc_std=(accuracy_std+accuracy_std_1)./2;
lssvm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lssvm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lssvm_spe=(specificity_mean+specificity_mean_1)./2;
lssvm_spe_std=(specificity_std+specificity_std_1)./2;
lssvm_pre=(precision_mean+precision_mean_1)./2;
lssvm_pre_std=(precision_std+precision_std_1)./2;
lssvm_npv=(npv_mean+npv_mean_1)./2;
lssvm_npv_std=(npv_std+npv_std_1)./2;
lssvm_F1=(F1_score_mean+F1_score_mean_1)./2;
lssvm_F1_std=(F1_score_std+F1_score_std_1)./2;
lssvm=[lssvm_acc lssvm_acc_std;lssvm_sen lssvm_sen_std;lssvm_spe lssvm_spe_std;lssvm_pre lssvm_pre_std;lssvm_npv lssvm_npv_std;lssvm_F1 lssvm_F1_std;];

%elm
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

elm_acc=(accuracy_mean+accuracy_mean_1)./2;
elm_acc_std=(accuracy_std+accuracy_std_1)./2;
elm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
elm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
elm_spe=(specificity_mean+specificity_mean_1)./2;
elm_spe_std=(specificity_std+specificity_std_1)./2;
elm_pre=(precision_mean+precision_mean_1)./2;
elm_pre_std=(precision_std+precision_std_1)./2;
elm_npv=(npv_mean+npv_mean_1)./2;
elm_npv_std=(npv_std+npv_std_1)./2;
elm_F1=(F1_score_mean+F1_score_mean_1)./2;
elm_F1_std=(F1_score_std+F1_score_std_1)./2;
elm=[elm_acc elm_acc_std;elm_sen elm_sen_std;elm_spe elm_spe_std;elm_pre elm_pre_std;elm_npv elm_npv_std;elm_F1 elm_F1_std;];

%nb
load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

nb_acc=(accuracy_mean+accuracy_mean_1)./2;
nb_acc_std=(accuracy_std+accuracy_std_1)./2;
nb_sen=(sensitivity_mean+sensitivity_mean_1)./2;
nb_sen_std=(sensitivity_std+sensitivity_std_1)./2;
nb_spe=(specificity_mean+specificity_mean_1)./2;
nb_spe_std=(specificity_std+specificity_std_1)./2;
nb_pre=(precision_mean+precision_mean_1)./2;
nb_pre_std=(precision_std+precision_std_1)./2;
nb_npv=(npv_mean+npv_mean_1)./2;
nb_npv_std=(npv_std+npv_std_1)./2;
nb_F1=(F1_score_mean+F1_score_mean_1)./2;
nb_F1_std=(F1_score_std+F1_score_std_1)./2;
nb=[nb_acc nb_acc_std;nb_sen nb_sen_std;nb_spe nb_spe_std;nb_pre nb_pre_std;nb_npv nb_npv_std;nb_F1 nb_F1_std;];

%lr
load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

lr_acc=(accuracy_mean+accuracy_mean_1)./2;
lr_acc_std=(accuracy_std+accuracy_std_1)./2;
lr_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lr_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lr_spe=(specificity_mean+specificity_mean_1)./2;
lr_spe_std=(specificity_std+specificity_std_1)./2;
lr_pre=(precision_mean+precision_mean_1)./2;
lr_pre_std=(precision_std+precision_std_1)./2;
lr_npv=(npv_mean+npv_mean_1)./2;
lr_npv_std=(npv_std+npv_std_1)./2;
lr_F1=(F1_score_mean+F1_score_mean_1)./2;
lr_F1_std=(F1_score_std+F1_score_std_1)./2;
lr=[lr_acc lr_acc_std;lr_sen lr_sen_std;lr_spe lr_spe_std;lr_pre lr_pre_std;lr_npv lr_npv_std;lr_F1 lr_F1_std;];

%knn
load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

knn_acc=(accuracy_mean+accuracy_mean_1)./2;
knn_acc_std=(accuracy_std+accuracy_std_1)./2;
knn_sen=(sensitivity_mean+sensitivity_mean_1)./2;
knn_sen_std=(sensitivity_std+sensitivity_std_1)./2;
knn_spe=(specificity_mean+specificity_mean_1)./2;
knn_spe_std=(specificity_std+specificity_std_1)./2;
knn_pre=(precision_mean+precision_mean_1)./2;
knn_pre_std=(precision_std+precision_std_1)./2;
knn_npv=(npv_mean+npv_mean_1)./2;
knn_npv_std=(npv_std+npv_std_1)./2;
knn_F1=(F1_score_mean+F1_score_mean_1)./2;
knn_F1_std=(F1_score_std+F1_score_std_1)./2;
knn=[knn_acc knn_acc_std;knn_sen knn_sen_std;knn_spe knn_spe_std;knn_pre knn_pre_std;knn_npv knn_npv_std;knn_F1 knn_F1_std;];

%ann
load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

ann_acc=(accuracy_mean+accuracy_mean_1)./2;
ann_acc_std=(accuracy_std+accuracy_std_1)./2;
ann_sen=(sensitivity_mean+sensitivity_mean_1)./2;
ann_sen_std=(sensitivity_std+sensitivity_std_1)./2;
ann_spe=(specificity_mean+specificity_mean_1)./2;
ann_spe_std=(specificity_std+specificity_std_1)./2;
ann_pre=(precision_mean+precision_mean_1)./2;
ann_pre_std=(precision_std+precision_std_1)./2;
ann_npv=(npv_mean+npv_mean_1)./2;
ann_npv_std=(npv_std+npv_std_1)./2;
ann_F1=(F1_score_mean+F1_score_mean_1)./2;
ann_F1_std=(F1_score_std+F1_score_std_1)./2;
ann=[ann_acc ann_acc_std;ann_sen ann_sen_std;ann_spe ann_spe_std;ann_pre ann_pre_std;ann_npv ann_npv_std;ann_F1 ann_F1_std;];

%lpp_lssvm
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lssvm_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lssvm_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

lpp_lssvm_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_lssvm_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_lssvm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_lssvm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_lssvm_spe=(specificity_mean+specificity_mean_1)./2;
lpp_lssvm_spe_std=(specificity_std+specificity_std_1)./2;
lpp_lssvm_pre=(precision_mean+precision_mean_1)./2;
lpp_lssvm_pre_std=(precision_std+precision_std_1)./2;
lpp_lssvm_npv=(npv_mean+npv_mean_1)./2;
lpp_lssvm_npv_std=(npv_std+npv_std_1)./2;
lpp_lssvm_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_lssvm_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_lssvm=[lpp_lssvm_acc lpp_lssvm_acc_std;lpp_lssvm_sen lpp_lssvm_sen_std;lpp_lssvm_spe lpp_lssvm_spe_std;lpp_lssvm_pre lpp_lssvm_pre_std;lpp_lssvm_npv lpp_lssvm_npv_std;lpp_lssvm_F1 lpp_lssvm_F1_std;];

%lpp_elm
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

lpp_elm_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_elm_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_elm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_elm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_elm_spe=(specificity_mean+specificity_mean_1)./2;
lpp_elm_spe_std=(specificity_std+specificity_std_1)./2;
lpp_elm_pre=(precision_mean+precision_mean_1)./2;
lpp_elm_pre_std=(precision_std+precision_std_1)./2;
lpp_elm_npv=(npv_mean+npv_mean_1)./2;
lpp_elm_npv_std=(npv_std+npv_std_1)./2;
lpp_elm_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_elm_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_elm=[lpp_elm_acc lpp_elm_acc_std;lpp_elm_sen lpp_elm_sen_std;lpp_elm_spe lpp_elm_spe_std;lpp_elm_pre lpp_elm_pre_std;lpp_elm_npv lpp_elm_npv_std;lpp_elm_F1 lpp_elm_F1_std;];

%lpp_nb
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_nb_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_nb_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

lpp_nb_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_nb_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_nb_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_nb_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_nb_spe=(specificity_mean+specificity_mean_1)./2;
lpp_nb_spe_std=(specificity_std+specificity_std_1)./2;
lpp_nb_pre=(precision_mean+precision_mean_1)./2;
lpp_nb_pre_std=(precision_std+precision_std_1)./2;
lpp_nb_npv=(npv_mean+npv_mean_1)./2;
lpp_nb_npv_std=(npv_std+npv_std_1)./2;
lpp_nb_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_nb_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_nb=[lpp_nb_acc lpp_nb_acc_std;lpp_nb_sen lpp_nb_sen_std;lpp_nb_spe lpp_nb_spe_std;lpp_nb_pre lpp_nb_pre_std;lpp_nb_npv lpp_nb_npv_std;lpp_nb_F1 lpp_nb_F1_std;];

%lpp_lr
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

lpp_lr_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_lr_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_lr_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_lr_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_lr_spe=(specificity_mean+specificity_mean_1)./2;
lpp_lr_spe_std=(specificity_std+specificity_std_1)./2;
lpp_lr_pre=(precision_mean+precision_mean_1)./2;
lpp_lr_pre_std=(precision_std+precision_std_1)./2;
lpp_lr_npv=(npv_mean+npv_mean_1)./2;
lpp_lr_npv_std=(npv_std+npv_std_1)./2;
lpp_lr_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_lr_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_lr=[lpp_lr_acc lpp_lr_acc_std;lpp_lr_sen lpp_lr_sen_std;lpp_lr_spe lpp_lr_spe_std;lpp_lr_pre lpp_lr_pre_std;lpp_lr_npv lpp_lr_npv_std;lpp_lr_F1 lpp_lr_F1_std;];

%lpp_knn
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_knn_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_knn_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

lpp_knn_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_knn_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_knn_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_knn_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_knn_spe=(specificity_mean+specificity_mean_1)./2;
lpp_knn_spe_std=(specificity_std+specificity_std_1)./2;
lpp_knn_pre=(precision_mean+precision_mean_1)./2;
lpp_knn_pre_std=(precision_std+precision_std_1)./2;
lpp_knn_npv=(npv_mean+npv_mean_1)./2;
lpp_knn_npv_std=(npv_std+npv_std_1)./2;
lpp_knn_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_knn_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_knn=[lpp_knn_acc lpp_knn_acc_std;lpp_knn_sen lpp_knn_sen_std;lpp_knn_spe lpp_knn_spe_std;lpp_knn_pre lpp_knn_pre_std;lpp_knn_npv lpp_knn_npv_std;lpp_knn_F1 lpp_knn_F1_std;];

%lpp_ann
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_ann_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_ann_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

lpp_ann_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_ann_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_ann_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_ann_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_ann_spe=(specificity_mean+specificity_mean_1)./2;
lpp_ann_spe_std=(specificity_std+specificity_std_1)./2;
lpp_ann_pre=(precision_mean+precision_mean_1)./2;
lpp_ann_pre_std=(precision_std+precision_std_1)./2;
lpp_ann_npv=(npv_mean+npv_mean_1)./2;
lpp_ann_npv_std=(npv_std+npv_std_1)./2;
lpp_ann_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_ann_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_ann=[lpp_ann_acc lpp_ann_acc_std;lpp_ann_sen lpp_ann_sen_std;lpp_ann_spe lpp_ann_spe_std;lpp_ann_pre lpp_ann_pre_std;lpp_ann_npv lpp_ann_npv_std;lpp_ann_F1 lpp_ann_F1_std;];

%h_elm
load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task1_session1
accuracy_mean=mean(per_index_single1(:,4));
accuracy_std=std(per_index_single1(:,4));
sensitivity_mean=mean(per_index_single1(:,6));
sensitivity_std=std(per_index_single1(:,6));
specificity_mean=mean(per_index_single1(:,7));
specificity_std=std(per_index_single1(:,7));
precision_mean=mean(per_index_single1(:,8));
precision_std=std(per_index_single1(:,8));
npv_mean=mean(per_index_single1(:,9));
npv_std=std(per_index_single1(:,9));
F1_score_mean=mean(per_index_single1(:,10));
F1_score_std=std(per_index_single1(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task1_session2
accuracy_mean_1=mean(per_index_single2(:,4));
accuracy_std_1=std(per_index_single2(:,4));
sensitivity_mean_1=mean(per_index_single2(:,6));
sensitivity_std_1=std(per_index_single2(:,6));
specificity_mean_1=mean(per_index_single2(:,7));
specificity_std_1=std(per_index_single2(:,7));
precision_mean_1=mean(per_index_single2(:,8));
precision_std_1=std(per_index_single2(:,8));
npv_mean_1=mean(per_index_single2(:,9));
npv_std_1=std(per_index_single2(:,9));
F1_score_mean_1=mean(per_index_single2(:,10));
F1_score_std_1=std(per_index_single2(:,10));

h_elm_acc=(accuracy_mean+accuracy_mean_1)./2;
h_elm_acc_std=(accuracy_std+accuracy_std_1)./2;
h_elm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
h_elm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
h_elm_spe=(specificity_mean+specificity_mean_1)./2;
h_elm_spe_std=(specificity_std+specificity_std_1)./2;
h_elm_pre=(precision_mean+precision_mean_1)./2;
h_elm_pre_std=(precision_std+precision_std_1)./2;
h_elm_npv=(npv_mean+npv_mean_1)./2;
h_elm_npv_std=(npv_std+npv_std_1)./2;
h_elm_F1=(F1_score_mean+F1_score_mean_1)./2;
h_elm_F1_std=(F1_score_std+F1_score_std_1)./2;
h_elm=[h_elm_acc h_elm_acc_std;h_elm_sen h_elm_sen_std;h_elm_spe h_elm_spe_std;h_elm_pre h_elm_pre_std;h_elm_npv h_elm_npv_std;h_elm_F1 h_elm_F1_std;];

%ED_ELM
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case1_result
ED_ELM=[case1_per_mean(1,4) case1_per_sd(1,4);case1_per_mean(1,5) case1_per_sd(1,5);case1_per_mean(1,6) case1_per_sd(1,6);case1_per_mean(1,7) case1_per_sd(1,7);case1_per_mean(1,8) case1_per_sd(1,8);case1_per_mean(1,9) case1_per_sd(1,9)];

% case1=[lssvm(1,1);elm(1,1);nb(1,1);lr(1,1);knn(1,1);ann(1,1);lpp_lssvm(1,1);lpp_elm(1,1);lpp_nb(1,1);lpp_lr(1,1);lpp_knn(1,1);lpp_ann(1,1);h_elm(1,1) ];
a=[lssvm(1,1) lssvm(2,1) lssvm(3,1) lssvm(4,1) lssvm(5,1) lssvm(6,1);elm(1,1) elm(2,1) elm(3,1) elm(4,1) elm(5,1) elm(6,1);nb(1,1) nb(2,1) nb(3,1) nb(4,1) nb(5,1) nb(6,1);lr(1,1) lr(2,1) lr(3,1) lr(4,1) lr(5,1) lr(6,1);
      knn(1,1) knn(2,1) knn(3,1) knn(4,1) knn(5,1) knn(6,1);ann(1,1) ann(2,1) ann(3,1) ann(4,1) ann(5,1) ann(6,1);lpp_lssvm(1,1) lpp_lssvm(2,1) lpp_lssvm(3,1) lpp_lssvm(4,1) lpp_lssvm(5,1) lpp_lssvm(6,1);lpp_elm(1,1) lpp_elm(2,1) lpp_elm(3,1) lpp_elm(4,1) lpp_elm(5,1) lpp_elm(6,1);
      lpp_nb(1,1) lpp_nb(2,1) lpp_nb(3,1) lpp_nb(4,1) lpp_nb(5,1) lpp_nb(6,1);lpp_lr(1,1) lpp_lr(2,1) lpp_lr(3,1) lpp_lr(4,1) lpp_lr(5,1) lpp_lr(6,1);lpp_knn(1,1) lpp_knn(2,1) lpp_knn(3,1) lpp_knn(4,1) lpp_knn(5,1) lpp_knn(6,1);lpp_ann(1,1) lpp_ann(2,1) lpp_ann(3,1) lpp_ann(4,1) lpp_ann(5,1) lpp_ann(6,1);
      h_elm(1,1) h_elm(2,1) h_elm(3,1) h_elm(4,1) h_elm(5,1) h_elm(6,1);ED_ELM(1,1) ED_ELM(2,1) ED_ELM(3,1) ED_ELM(4,1) ED_ELM(5,1) ED_ELM(6,1)];
b=[lssvm(1,2) lssvm(2,2) lssvm(3,2) lssvm(4,2) lssvm(5,2) lssvm(6,2);elm(1,2) elm(2,2) elm(3,2) elm(4,2) elm(5,2) elm(6,2);nb(1,2) nb(2,2) nb(3,2) nb(4,2) nb(5,2) nb(6,2);lr(1,2) lr(2,2) lr(3,2) lr(4,2) lr(5,2) lr(6,2);
      knn(1,2) knn(2,2) knn(3,2) knn(4,2) knn(5,2) knn(6,2);ann(1,2) ann(2,2) ann(3,2) ann(4,2) ann(5,2) ann(6,2);lpp_lssvm(1,2) lpp_lssvm(2,2) lpp_lssvm(3,2) lpp_lssvm(4,2) lpp_lssvm(5,2) lpp_lssvm(6,2);lpp_elm(1,2) lpp_elm(2,2) lpp_elm(3,2) lpp_elm(4,2) lpp_elm(5,2) lpp_elm(6,2);
      lpp_nb(1,2) lpp_nb(2,2) lpp_nb(3,2) lpp_nb(4,2) lpp_nb(5,2) lpp_nb(6,2);lpp_lr(1,2) lpp_lr(2,2) lpp_lr(3,2) lpp_lr(4,2) lpp_lr(5,2) lpp_lr(6,2);lpp_knn(1,2) lpp_knn(2,2) lpp_knn(3,2) lpp_knn(4,2) lpp_knn(5,2) lpp_knn(6,2);lpp_ann(1,2) lpp_ann(2,2) lpp_ann(3,2) lpp_ann(4,2) lpp_ann(5,2) lpp_ann(6,2);
      h_elm(1,2) h_elm(2,2) h_elm(3,2) h_elm(4,2) h_elm(5,2) h_elm(6,2);ED_ELM(1,2) ED_ELM(2,2) ED_ELM(3,2) ED_ELM(4,2) ED_ELM(5,2) ED_ELM(6,2)];
case1=[a b];
save F:\matlab\trial_procedure\study_1\fig10_11\case1 case1


%case2
%lssvm
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lssvm_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

lssvm_acc=(accuracy_mean+accuracy_mean_1)./2;
lssvm_acc_std=(accuracy_std+accuracy_std_1)./2;
lssvm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lssvm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lssvm_spe=(specificity_mean+specificity_mean_1)./2;
lssvm_spe_std=(specificity_std+specificity_std_1)./2;
lssvm_pre=(precision_mean+precision_mean_1)./2;
lssvm_pre_std=(precision_std+precision_std_1)./2;
lssvm_npv=(npv_mean+npv_mean_1)./2;
lssvm_npv_std=(npv_std+npv_std_1)./2;
lssvm_F1=(F1_score_mean+F1_score_mean_1)./2;
lssvm_F1_std=(F1_score_std+F1_score_std_1)./2;
lssvm=[lssvm_acc lssvm_acc_std;lssvm_sen lssvm_sen_std;lssvm_spe lssvm_spe_std;lssvm_pre lssvm_pre_std;lssvm_npv lssvm_npv_std;lssvm_F1 lssvm_F1_std;];

%elm
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\elm_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

elm_acc=(accuracy_mean+accuracy_mean_1)./2;
elm_acc_std=(accuracy_std+accuracy_std_1)./2;
elm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
elm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
elm_spe=(specificity_mean+specificity_mean_1)./2;
elm_spe_std=(specificity_std+specificity_std_1)./2;
elm_pre=(precision_mean+precision_mean_1)./2;
elm_pre_std=(precision_std+precision_std_1)./2;
elm_npv=(npv_mean+npv_mean_1)./2;
elm_npv_std=(npv_std+npv_std_1)./2;
elm_F1=(F1_score_mean+F1_score_mean_1)./2;
elm_F1_std=(F1_score_std+F1_score_std_1)./2;
elm=[elm_acc elm_acc_std;elm_sen elm_sen_std;elm_spe elm_spe_std;elm_pre elm_pre_std;elm_npv elm_npv_std;elm_F1 elm_F1_std;];

%nb
load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\nb_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

nb_acc=(accuracy_mean+accuracy_mean_1)./2;
nb_acc_std=(accuracy_std+accuracy_std_1)./2;
nb_sen=(sensitivity_mean+sensitivity_mean_1)./2;
nb_sen_std=(sensitivity_std+sensitivity_std_1)./2;
nb_spe=(specificity_mean+specificity_mean_1)./2;
nb_spe_std=(specificity_std+specificity_std_1)./2;
nb_pre=(precision_mean+precision_mean_1)./2;
nb_pre_std=(precision_std+precision_std_1)./2;
nb_npv=(npv_mean+npv_mean_1)./2;
nb_npv_std=(npv_std+npv_std_1)./2;
nb_F1=(F1_score_mean+F1_score_mean_1)./2;
nb_F1_std=(F1_score_std+F1_score_std_1)./2;
nb=[nb_acc nb_acc_std;nb_sen nb_sen_std;nb_spe nb_spe_std;nb_pre nb_pre_std;nb_npv nb_npv_std;nb_F1 nb_F1_std;];

%lr
load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lr_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

lr_acc=(accuracy_mean+accuracy_mean_1)./2;
lr_acc_std=(accuracy_std+accuracy_std_1)./2;
lr_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lr_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lr_spe=(specificity_mean+specificity_mean_1)./2;
lr_spe_std=(specificity_std+specificity_std_1)./2;
lr_pre=(precision_mean+precision_mean_1)./2;
lr_pre_std=(precision_std+precision_std_1)./2;
lr_npv=(npv_mean+npv_mean_1)./2;
lr_npv_std=(npv_std+npv_std_1)./2;
lr_F1=(F1_score_mean+F1_score_mean_1)./2;
lr_F1_std=(F1_score_std+F1_score_std_1)./2;
lr=[lr_acc lr_acc_std;lr_sen lr_sen_std;lr_spe lr_spe_std;lr_pre lr_pre_std;lr_npv lr_npv_std;lr_F1 lr_F1_std;];

%knn
load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\knn_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

knn_acc=(accuracy_mean+accuracy_mean_1)./2;
knn_acc_std=(accuracy_std+accuracy_std_1)./2;
knn_sen=(sensitivity_mean+sensitivity_mean_1)./2;
knn_sen_std=(sensitivity_std+sensitivity_std_1)./2;
knn_spe=(specificity_mean+specificity_mean_1)./2;
knn_spe_std=(specificity_std+specificity_std_1)./2;
knn_pre=(precision_mean+precision_mean_1)./2;
knn_pre_std=(precision_std+precision_std_1)./2;
knn_npv=(npv_mean+npv_mean_1)./2;
knn_npv_std=(npv_std+npv_std_1)./2;
knn_F1=(F1_score_mean+F1_score_mean_1)./2;
knn_F1_std=(F1_score_std+F1_score_std_1)./2;
knn=[knn_acc knn_acc_std;knn_sen knn_sen_std;knn_spe knn_spe_std;knn_pre knn_pre_std;knn_npv knn_npv_std;knn_F1 knn_F1_std;];

%ann
load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\ann_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

ann_acc=(accuracy_mean+accuracy_mean_1)./2;
ann_acc_std=(accuracy_std+accuracy_std_1)./2;
ann_sen=(sensitivity_mean+sensitivity_mean_1)./2;
ann_sen_std=(sensitivity_std+sensitivity_std_1)./2;
ann_spe=(specificity_mean+specificity_mean_1)./2;
ann_spe_std=(specificity_std+specificity_std_1)./2;
ann_pre=(precision_mean+precision_mean_1)./2;
ann_pre_std=(precision_std+precision_std_1)./2;
ann_npv=(npv_mean+npv_mean_1)./2;
ann_npv_std=(npv_std+npv_std_1)./2;
ann_F1=(F1_score_mean+F1_score_mean_1)./2;
ann_F1_std=(F1_score_std+F1_score_std_1)./2;
ann=[ann_acc ann_acc_std;ann_sen ann_sen_std;ann_spe ann_spe_std;ann_pre ann_pre_std;ann_npv ann_npv_std;ann_F1 ann_F1_std;];

%lpp_lssvm
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lssvm_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lssvm_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

lpp_lssvm_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_lssvm_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_lssvm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_lssvm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_lssvm_spe=(specificity_mean+specificity_mean_1)./2;
lpp_lssvm_spe_std=(specificity_std+specificity_std_1)./2;
lpp_lssvm_pre=(precision_mean+precision_mean_1)./2;
lpp_lssvm_pre_std=(precision_std+precision_std_1)./2;
lpp_lssvm_npv=(npv_mean+npv_mean_1)./2;
lpp_lssvm_npv_std=(npv_std+npv_std_1)./2;
lpp_lssvm_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_lssvm_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_lssvm=[lpp_lssvm_acc lpp_lssvm_acc_std;lpp_lssvm_sen lpp_lssvm_sen_std;lpp_lssvm_spe lpp_lssvm_spe_std;lpp_lssvm_pre lpp_lssvm_pre_std;lpp_lssvm_npv lpp_lssvm_npv_std;lpp_lssvm_F1 lpp_lssvm_F1_std;];

%lpp_elm
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_elm_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

lpp_elm_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_elm_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_elm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_elm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_elm_spe=(specificity_mean+specificity_mean_1)./2;
lpp_elm_spe_std=(specificity_std+specificity_std_1)./2;
lpp_elm_pre=(precision_mean+precision_mean_1)./2;
lpp_elm_pre_std=(precision_std+precision_std_1)./2;
lpp_elm_npv=(npv_mean+npv_mean_1)./2;
lpp_elm_npv_std=(npv_std+npv_std_1)./2;
lpp_elm_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_elm_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_elm=[lpp_elm_acc lpp_elm_acc_std;lpp_elm_sen lpp_elm_sen_std;lpp_elm_spe lpp_elm_spe_std;lpp_elm_pre lpp_elm_pre_std;lpp_elm_npv lpp_elm_npv_std;lpp_elm_F1 lpp_elm_F1_std;];

%lpp_nb
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_nb_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_nb_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

lpp_nb_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_nb_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_nb_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_nb_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_nb_spe=(specificity_mean+specificity_mean_1)./2;
lpp_nb_spe_std=(specificity_std+specificity_std_1)./2;
lpp_nb_pre=(precision_mean+precision_mean_1)./2;
lpp_nb_pre_std=(precision_std+precision_std_1)./2;
lpp_nb_npv=(npv_mean+npv_mean_1)./2;
lpp_nb_npv_std=(npv_std+npv_std_1)./2;
lpp_nb_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_nb_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_nb=[lpp_nb_acc lpp_nb_acc_std;lpp_nb_sen lpp_nb_sen_std;lpp_nb_spe lpp_nb_spe_std;lpp_nb_pre lpp_nb_pre_std;lpp_nb_npv lpp_nb_npv_std;lpp_nb_F1 lpp_nb_F1_std;];

%lpp_lr
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_lr_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

lpp_lr_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_lr_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_lr_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_lr_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_lr_spe=(specificity_mean+specificity_mean_1)./2;
lpp_lr_spe_std=(specificity_std+specificity_std_1)./2;
lpp_lr_pre=(precision_mean+precision_mean_1)./2;
lpp_lr_pre_std=(precision_std+precision_std_1)./2;
lpp_lr_npv=(npv_mean+npv_mean_1)./2;
lpp_lr_npv_std=(npv_std+npv_std_1)./2;
lpp_lr_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_lr_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_lr=[lpp_lr_acc lpp_lr_acc_std;lpp_lr_sen lpp_lr_sen_std;lpp_lr_spe lpp_lr_spe_std;lpp_lr_pre lpp_lr_pre_std;lpp_lr_npv lpp_lr_npv_std;lpp_lr_F1 lpp_lr_F1_std;];

%lpp_knn
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_knn_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_knn_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

lpp_knn_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_knn_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_knn_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_knn_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_knn_spe=(specificity_mean+specificity_mean_1)./2;
lpp_knn_spe_std=(specificity_std+specificity_std_1)./2;
lpp_knn_pre=(precision_mean+precision_mean_1)./2;
lpp_knn_pre_std=(precision_std+precision_std_1)./2;
lpp_knn_npv=(npv_mean+npv_mean_1)./2;
lpp_knn_npv_std=(npv_std+npv_std_1)./2;
lpp_knn_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_knn_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_knn=[lpp_knn_acc lpp_knn_acc_std;lpp_knn_sen lpp_knn_sen_std;lpp_knn_spe lpp_knn_spe_std;lpp_knn_pre lpp_knn_pre_std;lpp_knn_npv lpp_knn_npv_std;lpp_knn_F1 lpp_knn_F1_std;];

%lpp_ann
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_ann_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\lpp_ann_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

lpp_ann_acc=(accuracy_mean+accuracy_mean_1)./2;
lpp_ann_acc_std=(accuracy_std+accuracy_std_1)./2;
lpp_ann_sen=(sensitivity_mean+sensitivity_mean_1)./2;
lpp_ann_sen_std=(sensitivity_std+sensitivity_std_1)./2;
lpp_ann_spe=(specificity_mean+specificity_mean_1)./2;
lpp_ann_spe_std=(specificity_std+specificity_std_1)./2;
lpp_ann_pre=(precision_mean+precision_mean_1)./2;
lpp_ann_pre_std=(precision_std+precision_std_1)./2;
lpp_ann_npv=(npv_mean+npv_mean_1)./2;
lpp_ann_npv_std=(npv_std+npv_std_1)./2;
lpp_ann_F1=(F1_score_mean+F1_score_mean_1)./2;
lpp_ann_F1_std=(F1_score_std+F1_score_std_1)./2;
lpp_ann=[lpp_ann_acc lpp_ann_acc_std;lpp_ann_sen lpp_ann_sen_std;lpp_ann_spe lpp_ann_spe_std;lpp_ann_pre lpp_ann_pre_std;lpp_ann_npv lpp_ann_npv_std;lpp_ann_F1 lpp_ann_F1_std;];

%h_elm
load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task2_session1
accuracy_mean=mean(per_index_single3(:,4));
accuracy_std=std(per_index_single3(:,4));
sensitivity_mean=mean(per_index_single3(:,6));
sensitivity_std=std(per_index_single3(:,6));
specificity_mean=mean(per_index_single3(:,7));
specificity_std=std(per_index_single3(:,7));
precision_mean=mean(per_index_single3(:,8));
precision_std=std(per_index_single3(:,8));
npv_mean=mean(per_index_single3(:,9));
npv_std=std(per_index_single3(:,9));
F1_score_mean=mean(per_index_single3(:,10));
F1_score_std=std(per_index_single3(:,10));
load F:\matlab\trial_procedure\study_1\data_analysis\h_elm_cross_subject\task2_session2
accuracy_mean_1=mean(per_index_single4(:,4));
accuracy_std_1=std(per_index_single4(:,4));
sensitivity_mean_1=mean(per_index_single4(:,6));
sensitivity_std_1=std(per_index_single4(:,6));
specificity_mean_1=mean(per_index_single4(:,7));
specificity_std_1=std(per_index_single4(:,7));
precision_mean_1=mean(per_index_single4(:,8));
precision_std_1=std(per_index_single4(:,8));
npv_mean_1=mean(per_index_single4(:,9));
npv_std_1=std(per_index_single4(:,9));
F1_score_mean_1=mean(per_index_single4(:,10));
F1_score_std_1=std(per_index_single4(:,10));

h_elm_acc=(accuracy_mean+accuracy_mean_1)./2;
h_elm_acc_std=(accuracy_std+accuracy_std_1)./2;
h_elm_sen=(sensitivity_mean+sensitivity_mean_1)./2;
h_elm_sen_std=(sensitivity_std+sensitivity_std_1)./2;
h_elm_spe=(specificity_mean+specificity_mean_1)./2;
h_elm_spe_std=(specificity_std+specificity_std_1)./2;
h_elm_pre=(precision_mean+precision_mean_1)./2;
h_elm_pre_std=(precision_std+precision_std_1)./2;
h_elm_npv=(npv_mean+npv_mean_1)./2;
h_elm_npv_std=(npv_std+npv_std_1)./2;
h_elm_F1=(F1_score_mean+F1_score_mean_1)./2;
h_elm_F1_std=(F1_score_std+F1_score_std_1)./2;
h_elm=[h_elm_acc h_elm_acc_std;h_elm_sen h_elm_sen_std;h_elm_spe h_elm_spe_std;h_elm_pre h_elm_pre_std;h_elm_npv h_elm_npv_std;h_elm_F1 h_elm_F1_std;];

%ED_ELM
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case2_result
ED_ELM=[case2_per_mean(1,4) case2_per_sd(1,4);case2_per_mean(1,5) case2_per_sd(1,5);case2_per_mean(1,6) case2_per_sd(1,6);case2_per_mean(1,7) case2_per_sd(1,7);case2_per_mean(1,8) case2_per_sd(1,8);case2_per_mean(1,9) case2_per_sd(1,9)];

% case2=[lssvm(1,1);elm(1,1);nb(1,1);lr(1,1);knn(1,1);ann(1,1);lpp_lssvm(1,1);lpp_elm(1,1);lpp_nb(1,1);lpp_lr(1,1);lpp_knn(1,1);lpp_ann(1,1);h_elm(1,1) ];
c=[lssvm(1,1) lssvm(2,1) lssvm(3,1) lssvm(4,1) lssvm(5,1) lssvm(6,1);elm(1,1) elm(2,1) elm(3,1) elm(4,1) elm(5,1) elm(6,1);nb(1,1) nb(2,1) nb(3,1) nb(4,1) nb(5,1) nb(6,1);lr(1,1) lr(2,1) lr(3,1) lr(4,1) lr(5,1) lr(6,1);
      knn(1,1) knn(2,1) knn(3,1) knn(4,1) knn(5,1) knn(6,1);ann(1,1) ann(2,1) ann(3,1) ann(4,1) ann(5,1) ann(6,1);lpp_lssvm(1,1) lpp_lssvm(2,1) lpp_lssvm(3,1) lpp_lssvm(4,1) lpp_lssvm(5,1) lpp_lssvm(6,1);lpp_elm(1,1) lpp_elm(2,1) lpp_elm(3,1) lpp_elm(4,1) lpp_elm(5,1) lpp_elm(6,1);
      lpp_nb(1,1) lpp_nb(2,1) lpp_nb(3,1) lpp_nb(4,1) lpp_nb(5,1) lpp_nb(6,1);lpp_lr(1,1) lpp_lr(2,1) lpp_lr(3,1) lpp_lr(4,1) lpp_lr(5,1) lpp_lr(6,1);lpp_knn(1,1) lpp_knn(2,1) lpp_knn(3,1) lpp_knn(4,1) lpp_knn(5,1) lpp_knn(6,1);lpp_ann(1,1) lpp_ann(2,1) lpp_ann(3,1) lpp_ann(4,1) lpp_ann(5,1) lpp_ann(6,1);
      h_elm(1,1) h_elm(2,1) h_elm(3,1) h_elm(4,1) h_elm(5,1) h_elm(6,1);ED_ELM(1,1) ED_ELM(2,1) ED_ELM(3,1) ED_ELM(4,1) ED_ELM(5,1) ED_ELM(6,1)];
d=[lssvm(1,2) lssvm(2,2) lssvm(3,2) lssvm(4,2) lssvm(5,2) lssvm(6,2);elm(1,2) elm(2,2) elm(3,2) elm(4,2) elm(5,2) elm(6,2);nb(1,2) nb(2,2) nb(3,2) nb(4,2) nb(5,2) nb(6,2);lr(1,2) lr(2,2) lr(3,2) lr(4,2) lr(5,2) lr(6,2);
      knn(1,2) knn(2,2) knn(3,2) knn(4,2) knn(5,2) knn(6,2);ann(1,2) ann(2,2) ann(3,2) ann(4,2) ann(5,2) ann(6,2);lpp_lssvm(1,2) lpp_lssvm(2,2) lpp_lssvm(3,2) lpp_lssvm(4,2) lpp_lssvm(5,2) lpp_lssvm(6,2);lpp_elm(1,2) lpp_elm(2,2) lpp_elm(3,2) lpp_elm(4,2) lpp_elm(5,2) lpp_elm(6,2);
      lpp_nb(1,2) lpp_nb(2,2) lpp_nb(3,2) lpp_nb(4,2) lpp_nb(5,2) lpp_nb(6,2);lpp_lr(1,2) lpp_lr(2,2) lpp_lr(3,2) lpp_lr(4,2) lpp_lr(5,2) lpp_lr(6,2);lpp_knn(1,2) lpp_knn(2,2) lpp_knn(3,2) lpp_knn(4,2) lpp_knn(5,2) lpp_knn(6,2);lpp_ann(1,2) lpp_ann(2,2) lpp_ann(3,2) lpp_ann(4,2) lpp_ann(5,2) lpp_ann(6,2);
      h_elm(1,2) h_elm(2,2) h_elm(3,2) h_elm(4,2) h_elm(5,2) h_elm(6,2);ED_ELM(1,2) ED_ELM(2,2) ED_ELM(3,2) ED_ELM(4,2) ED_ELM(5,2) ED_ELM(6,2)];
case2=[c d];
save F:\matlab\trial_procedure\study_1\fig10_11\case2 case2

%fig10
%case1
subplot(3,2,1);
% bar(case1(:,1),'DisplayName','case1(:,1)'); 
bar(1,case1(1,1));hold on;
set(gca,'XTick',1:1:14); 
set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
bar(2,case1(2,1));hold on;
bar(3,case1(3,1));hold on;
bar(4,case1(4,1));hold on;
bar(5,case1(5,1));hold on;
bar(6,case1(6,1));hold on;
bar(7,case1(7,1));hold on;
bar(8,case1(8,1));hold on;
bar(9,case1(9,1));hold on;
bar(10,case1(10,1));hold on;
bar(11,case1(11,1));hold on;
bar(12,case1(12,1));hold on;
bar(13,case1(13,1));hold on;
bar(14,case1(14,1));hold on;
xtickangle(50);
% set(gca,'YTick',0:0.05:0.5);
ylabel('Accuracy','FontWeight','bold');
title('(a)','FontWeight','bold');
grid on;
hold on;
x = 1:1:14;
y=case1(:,1);
err=case1(:,7);
errorbar(x,y,err);

subplot(3,2,2);
% bar(case1(:,1),'DisplayName','case1(:,1)'); 
bar(1,case1(1,2));hold on;
set(gca,'XTick',1:1:14); 
set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
bar(2,case1(2,2));hold on;
bar(3,case1(3,2));hold on;
bar(4,case1(4,2));hold on;
bar(5,case1(5,2));hold on;
bar(6,case1(6,2));hold on;
bar(7,case1(7,2));hold on;
bar(8,case1(8,2));hold on;
bar(9,case1(9,2));hold on;
bar(10,case1(10,2));hold on;
bar(11,case1(11,2));hold on;
bar(12,case1(12,2));hold on;
bar(13,case1(13,2));hold on;
bar(14,case1(14,2));hold on;
xtickangle(50);
% set(gca,'YTick',0:0.05:0.5);
ylabel('Sensitivity','FontWeight','bold');
title('(b)','FontWeight','bold');
grid on;
hold on;
x = 1:1:14;
y=case1(:,2);
err=case1(:,8);
errorbar(x,y,err);

subplot(3,2,3);
% bar(case1(:,1),'DisplayName','case1(:,1)'); 
bar(1,case1(1,3));hold on;
set(gca,'XTick',1:1:14); 
set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
bar(2,case1(2,3));hold on;
bar(3,case1(3,3));hold on;
bar(4,case1(4,3));hold on;
bar(5,case1(5,3));hold on;
bar(6,case1(6,3));hold on;
bar(7,case1(7,3));hold on;
bar(8,case1(8,3));hold on;
bar(9,case1(9,3));hold on;
bar(10,case1(10,3));hold on;
bar(11,case1(11,3));hold on;
bar(12,case1(12,3));hold on;
bar(13,case1(13,3));hold on;
bar(14,case1(14,3));hold on;
xtickangle(50);
% set(gca,'YTick',0:0.05:0.5);
ylabel('Specificity','FontWeight','bold');
title('(c)','FontWeight','bold');
grid on;
hold on;
x = 1:1:14;
y=case1(:,3);
err=case1(:,9);
errorbar(x,y,err);

subplot(3,2,4);
% bar(case1(:,1),'DisplayName','case1(:,1)'); 
bar(1,case1(1,4));hold on;
set(gca,'XTick',1:1:14); 
set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
bar(2,case1(2,4));hold on;
bar(3,case1(3,4));hold on;
bar(4,case1(4,4));hold on;
bar(5,case1(5,4));hold on;
bar(6,case1(6,4));hold on;
bar(7,case1(7,4));hold on;
bar(8,case1(8,4));hold on;
bar(9,case1(9,4));hold on;
bar(10,case1(10,4));hold on;
bar(11,case1(11,4));hold on;
bar(12,case1(12,4));hold on;
bar(13,case1(13,4));hold on;
bar(14,case1(14,4));hold on;
xtickangle(50);
% set(gca,'YTick',0:0.05:0.5);
ylabel('Precision','FontWeight','bold');
title('(d)','FontWeight','bold');
grid on;
hold on;
x = 1:1:14;
y=case1(:,4);
err=case1(:,10);
errorbar(x,y,err);

subplot(3,2,5);
% bar(case1(:,1),'DisplayName','case1(:,1)'); 
bar(1,case1(1,5));hold on;
set(gca,'XTick',1:1:14); 
set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
bar(2,case1(2,5));hold on;
bar(3,case1(3,5));hold on;
bar(4,case1(4,5));hold on;
bar(5,case1(5,5));hold on;
bar(6,case1(6,5));hold on;
bar(7,case1(7,5));hold on;
bar(8,case1(8,5));hold on;
bar(9,case1(9,5));hold on;
bar(10,case1(10,5));hold on;
bar(11,case1(11,5));hold on;
bar(12,case1(12,5));hold on;
bar(13,case1(13,5));hold on;
bar(14,case1(14,5));hold on;
xtickangle(50);
% set(gca,'YTick',0:0.05:0.5);
ylabel('NPV','FontWeight','bold');
title('(e)','FontWeight','bold');
grid on;
hold on;
x = 1:1:14;
y=case1(:,5);
err=case1(:,11);
errorbar(x,y,err);

subplot(3,2,6);
% bar(case1(:,1),'DisplayName','case1(:,1)'); 
bar(1,case1(1,6));hold on;
set(gca,'XTick',1:1:14); 
set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
bar(2,case1(2,6));hold on;
bar(3,case1(3,6));hold on;
bar(4,case1(4,6));hold on;
bar(5,case1(5,6));hold on;
bar(6,case1(6,6));hold on;
bar(7,case1(7,6));hold on;
bar(8,case1(8,6));hold on;
bar(9,case1(9,6));hold on;
bar(10,case1(10,6));hold on;
bar(11,case1(11,6));hold on;
bar(12,case1(12,6));hold on;
bar(13,case1(13,6));hold on;
bar(14,case1(14,6));hold on;
xtickangle(50);
% set(gca,'YTick',0:0.05:0.5);
ylabel('F1-score','FontWeight','bold');
title('(f)','FontWeight','bold');
grid on;
hold on;
x = 1:1:14;
y=case1(:,6);
err=case1(:,12);
errorbar(x,y,err);



% %fig11
% %case2
% subplot(3,2,1);
% % bar(case1(:,1),'DisplayName','case1(:,1)'); 
% bar(1,case2(1,1));hold on;
% set(gca,'XTick',1:1:14); 
% set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
% bar(2,case2(2,1));hold on;
% bar(3,case2(3,1));hold on;
% bar(4,case2(4,1));hold on;
% bar(5,case2(5,1));hold on;
% bar(6,case2(6,1));hold on;
% bar(7,case2(7,1));hold on;
% bar(8,case2(8,1));hold on;
% bar(9,case2(9,1));hold on;
% bar(10,case2(10,1));hold on;
% bar(11,case2(11,1));hold on;
% bar(12,case2(12,1));hold on;
% bar(13,case2(13,1));hold on;
% bar(14,case2(14,1));hold on;
% xtickangle(50);
% % set(gca,'YTick',0:0.05:0.5);
% ylabel('Accuracy','FontWeight','bold');
% title('(a)','FontWeight','bold');
% grid on;
% hold on;
% x = 1:1:14;
% y=case2(:,1);
% err=case2(:,7);
% errorbar(x,y,err);
% 
% subplot(3,2,2);
% bar(1,case2(1,2));hold on;
% set(gca,'XTick',1:1:14); 
% set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
% bar(2,case2(2,2));hold on;
% bar(3,case2(3,2));hold on;
% bar(4,case2(4,2));hold on;
% bar(5,case2(5,2));hold on;
% bar(6,case2(6,2));hold on;
% bar(7,case2(7,2));hold on;
% bar(8,case2(8,2));hold on;
% bar(9,case2(9,2));hold on;
% bar(10,case2(10,2));hold on;
% bar(11,case2(11,2));hold on;
% bar(12,case2(12,2));hold on;
% bar(13,case2(13,2));hold on;
% bar(14,case2(14,2));hold on;
% xtickangle(50);
% % set(gca,'YTick',0:0.05:0.5);
% ylabel('Sensitivity','FontWeight','bold');
% title('(b)','FontWeight','bold');
% grid on;
% hold on;
% x = 1:1:14;
% y=case2(:,2);
% err=case2(:,8);
% errorbar(x,y,err);
% 
% subplot(3,2,3);
% bar(1,case2(1,3));hold on;
% set(gca,'XTick',1:1:14); 
% set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
% bar(2,case2(2,3));hold on;
% bar(3,case2(3,3));hold on;
% bar(4,case2(4,3));hold on;
% bar(5,case2(5,3));hold on;
% bar(6,case2(6,3));hold on;
% bar(7,case2(7,3));hold on;
% bar(8,case2(8,3));hold on;
% bar(9,case2(9,3));hold on;
% bar(10,case2(10,3));hold on;
% bar(11,case2(11,3));hold on;
% bar(12,case2(12,3));hold on;
% bar(13,case2(13,3));hold on;
% bar(14,case2(14,3));hold on;
% xtickangle(50);
% % set(gca,'YTick',0:0.05:0.5);
% ylabel('Specificity','FontWeight','bold');
% title('(c)','FontWeight','bold');
% grid on;
% hold on;
% x = 1:1:14;
% y=case2(:,3);
% err=case2(:,9);
% errorbar(x,y,err);
% 
% subplot(3,2,4);
% bar(1,case2(1,4));hold on;
% set(gca,'XTick',1:1:14); 
% set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
% bar(2,case2(2,4));hold on;
% bar(3,case2(3,4));hold on;
% bar(4,case2(4,4));hold on;
% bar(5,case2(5,4));hold on;
% bar(6,case2(6,4));hold on;
% bar(7,case2(7,4));hold on;
% bar(8,case2(8,4));hold on;
% bar(9,case2(9,4));hold on;
% bar(10,case2(10,4));hold on;
% bar(11,case2(11,4));hold on;
% bar(12,case2(12,4));hold on;
% bar(13,case2(13,4));hold on;
% bar(14,case2(14,4));hold on;
% xtickangle(50);
% % set(gca,'YTick',0:0.05:0.5);
% ylabel('Precision','FontWeight','bold');
% title('(d)','FontWeight','bold');
% grid on;
% hold on;
% x = 1:1:14;
% y=case2(:,4);
% err=case2(:,10);
% errorbar(x,y,err);
% 
% subplot(3,2,5);
% bar(1,case2(1,5));hold on;
% set(gca,'XTick',1:1:14); 
% set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
% bar(2,case2(2,5));hold on;
% bar(3,case2(3,5));hold on;
% bar(4,case2(4,5));hold on;
% bar(5,case2(5,5));hold on;
% bar(6,case2(6,5));hold on;
% bar(7,case2(7,5));hold on;
% bar(8,case2(8,5));hold on;
% bar(9,case2(9,5));hold on;
% bar(10,case2(10,5));hold on;
% bar(11,case2(11,5));hold on;
% bar(12,case2(12,5));hold on;
% bar(13,case2(13,5));hold on;
% bar(14,case2(14,5));hold on;
% xtickangle(50);
% % set(gca,'YTick',0:0.05:0.5);
% ylabel('NPV','FontWeight','bold');
% title('(e)','FontWeight','bold');
% grid on;
% hold on;
% x = 1:1:14;
% y=case2(:,5);
% err=case2(:,11);
% errorbar(x,y,err);
% 
% subplot(3,2,6);
% bar(1,case2(1,6));hold on;
% set(gca,'XTick',1:1:14); 
% set(gca,'XTickLabel',{'LSSVM','ELM','NB','LR','KNN','ANN','LPP-LSSVM','LPP-ELM','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','H-ELM','ED-ELM'});
% bar(2,case2(2,6));hold on;
% bar(3,case2(3,6));hold on;
% bar(4,case2(4,6));hold on;
% bar(5,case2(5,6));hold on;
% bar(6,case2(6,6));hold on;
% bar(7,case2(7,6));hold on;
% bar(8,case2(8,6));hold on;
% bar(9,case2(9,6));hold on;
% bar(10,case2(10,6));hold on;
% bar(11,case2(11,6));hold on;
% bar(12,case2(12,6));hold on;
% bar(13,case2(13,6));hold on;
% bar(14,case2(14,6));hold on;
% xtickangle(50);
% % set(gca,'YTick',0:0.05:0.5);
% ylabel('F1-score','FontWeight','bold');
% title('(f)','FontWeight','bold');
% grid on;
% hold on;
% x = 1:1:14;
% y=case2(:,6);
% err=case2(:,12);
% errorbar(x,y,err);