clc;
clear;
close all;
warning off;

%case1数据
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case1
nb(1,1)=acc_nb_case1;
lr(1,1)=acc_lr_case1;
knn(1,1)=acc_knn_case1;
ann(1,1)=acc_ann_case1;
elm(1,1)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case1_1
lssvm(1,1)=acc_lssvm_case1;
sae(1,1)=acc_sae_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case1_lpp
lpp_nb(1,1)=acc_nb_case1;
lpp_lr(1,1)=acc_lr_case1;
lpp_knn(1,1)=acc_knn_case1;
lpp_ann(1,1)=acc_ann_case1;
lpp_elm(1,1)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case1_1_lpp
lpp_lssvm(1,1)=acc_lssvm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case1_h_elm
h_elm(1,1)=acc_h_elm_case1;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case1
nb(1,2)=acc_nb_case1;
lr(1,2)=acc_lr_case1;
knn(1,2)=acc_knn_case1;
ann(1,2)=acc_ann_case1;
elm(1,2)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case1_1
lssvm(1,2)=acc_lssvm_case1;
sae(1,2)=acc_sae_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case1_lpp
lpp_nb(1,2)=acc_nb_case1;
lpp_lr(1,2)=acc_lr_case1;
lpp_knn(1,2)=acc_knn_case1;
lpp_ann(1,2)=acc_ann_case1;
lpp_elm(1,2)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case1_1_lpp
lpp_lssvm(1,2)=acc_lssvm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case1_h_elm
h_elm(1,2)=acc_h_elm_case1;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case1
nb(1,3)=acc_nb_case1;
lr(1,3)=acc_lr_case1;
knn(1,3)=acc_knn_case1;
ann(1,3)=acc_ann_case1;
elm(1,3)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case1_1
lssvm(1,3)=acc_lssvm_case1;
sae(1,3)=acc_sae_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case1_lpp
lpp_nb(1,3)=acc_nb_case1;
lpp_lr(1,3)=acc_lr_case1;
lpp_knn(1,3)=acc_knn_case1;
lpp_ann(1,3)=acc_ann_case1;
lpp_elm(1,3)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case1_1_lpp
lpp_lssvm(1,3)=acc_lssvm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case1_h_elm
h_elm(1,3)=acc_h_elm_case1;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case1
nb(1,4)=acc_nb_case1;
lr(1,4)=acc_lr_case1;
knn(1,4)=acc_knn_case1;
ann(1,4)=acc_ann_case1;
elm(1,4)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case1_1
lssvm(1,4)=acc_lssvm_case1;
sae(1,4)=acc_sae_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case1_lpp
lpp_nb(1,4)=acc_nb_case1;
lpp_lr(1,4)=acc_lr_case1;
lpp_knn(1,4)=acc_knn_case1;
lpp_ann(1,4)=acc_ann_case1;
lpp_elm(1,4)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case1_1_lpp
lpp_lssvm(1,4)=acc_lssvm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case1_h_elm
h_elm(1,4)=acc_h_elm_case1;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case1
nb(1,5)=acc_nb_case1;
lr(1,5)=acc_lr_case1;
knn(1,5)=acc_knn_case1;
ann(1,5)=acc_ann_case1;
elm(1,5)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case1_1
lssvm(1,5)=acc_lssvm_case1;
sae(1,5)=acc_sae_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case1_lpp
lpp_nb(1,5)=acc_nb_case1;
lpp_lr(1,5)=acc_lr_case1;
lpp_knn(1,5)=acc_knn_case1;
lpp_ann(1,5)=acc_ann_case1;
lpp_elm(1,5)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case1_1_lpp
lpp_lssvm(1,5)=acc_lssvm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case1_h_elm
h_elm(1,5)=acc_h_elm_case1;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject7_case1
nb(1,6)=acc_nb_case1;
lr(1,6)=acc_lr_case1;
knn(1,6)=acc_knn_case1;
ann(1,6)=acc_ann_case1;
elm(1,6)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject7_case1_1
lssvm(1,6)=acc_lssvm_case1;
sae(1,6)=acc_sae_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject7_case1_lpp
lpp_nb(1,6)=acc_nb_case1;
lpp_lr(1,6)=acc_lr_case1;
lpp_knn(1,6)=acc_knn_case1;
lpp_ann(1,6)=acc_ann_case1;
lpp_elm(1,6)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject7_case1_1_lpp
lpp_lssvm(1,6)=acc_lssvm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject7_case1_h_elm
h_elm(1,6)=acc_h_elm_case1;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject8_case1
nb(1,7)=acc_nb_case1;
lr(1,7)=acc_lr_case1;
knn(1,7)=acc_knn_case1;
ann(1,7)=acc_ann_case1;
elm(1,7)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject8_case1_1
lssvm(1,7)=acc_lssvm_case1;
sae(1,7)=acc_sae_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject8_case1_lpp
lpp_nb(1,7)=acc_nb_case1;
lpp_lr(1,7)=acc_lr_case1;
lpp_knn(1,7)=acc_knn_case1;
lpp_ann(1,7)=acc_ann_case1;
lpp_elm(1,7)=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject8_case1_1_lpp
lpp_lssvm(1,7)=acc_lssvm_case1;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject8_case1_h_elm
h_elm(1,7)=acc_h_elm_case1;
save F:\matlab\trial_procedure\study_1\number_of_subjects\case1 nb lr knn ann elm lssvm sae lpp_nb lpp_lr lpp_knn lpp_ann lpp_elm lpp_lssvm h_elm

%case2数据
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case2
nb_1(1,1)=acc_nb_case2;
lr_1(1,1)=acc_lr_case2;
knn_1(1,1)=acc_knn_case2;
ann_1(1,1)=acc_ann_case2;
elm_1(1,1)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case2_1
lssvm_1(1,1)=acc_lssvm_case2;
sae_1(1,1)=acc_sae_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case2_lpp
lpp_nb_1(1,1)=acc_nb_case2;
lpp_lr_1(1,1)=acc_lr_case2;
lpp_knn_1(1,1)=acc_knn_case2;
lpp_ann_1(1,1)=acc_ann_case2;
lpp_elm_1(1,1)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case2_1_lpp
lpp_lssvm_1(1,1)=acc_lssvm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case2_h_elm
h_elm_1(1,1)=acc_h_elm_case2;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case2
nb_1(1,2)=acc_nb_case2;
lr_1(1,2)=acc_lr_case2;
knn_1(1,2)=acc_knn_case2;
ann_1(1,2)=acc_ann_case2;
elm_1(1,2)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case2_1
lssvm_1(1,2)=acc_lssvm_case2;
sae_1(1,2)=acc_sae_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case2_lpp
lpp_nb_1(1,2)=acc_nb_case2;
lpp_lr_1(1,2)=acc_lr_case2;
lpp_knn_1(1,2)=acc_knn_case2;
lpp_ann_1(1,2)=acc_ann_case2;
lpp_elm_1(1,2)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case2_1_lpp
lpp_lssvm_1(1,2)=acc_lssvm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case2_h_elm
h_elm_1(1,2)=acc_h_elm_case2;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case2
nb_1(1,3)=acc_nb_case2;
lr_1(1,3)=acc_lr_case2;
knn_1(1,3)=acc_knn_case2;
ann_1(1,3)=acc_ann_case2;
elm_1(1,3)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case2_1
lssvm_1(1,3)=acc_lssvm_case2;
sae_1(1,3)=acc_sae_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case2_lpp
lpp_nb_1(1,3)=acc_nb_case2;
lpp_lr_1(1,3)=acc_lr_case2;
lpp_knn_1(1,3)=acc_knn_case2;
lpp_ann_1(1,3)=acc_ann_case2;
lpp_elm_1(1,3)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case2_1_lpp
lpp_lssvm_1(1,3)=acc_lssvm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case2_h_elm
h_elm_1(1,3)=acc_h_elm_case2;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case2
nb_1(1,4)=acc_nb_case2;
lr_1(1,4)=acc_lr_case2;
knn_1(1,4)=acc_knn_case2;
ann_1(1,4)=acc_ann_case2;
elm_1(1,4)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case2_1
lssvm_1(1,4)=acc_lssvm_case2;
sae_1(1,4)=acc_sae_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case2_lpp
lpp_nb_1(1,4)=acc_nb_case2;
lpp_lr_1(1,4)=acc_lr_case2;
lpp_knn_1(1,4)=acc_knn_case2;
lpp_ann_1(1,4)=acc_ann_case2;
lpp_elm_1(1,4)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case2_1_lpp
lpp_lssvm_1(1,4)=acc_lssvm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case2_h_elm
h_elm_1(1,4)=acc_h_elm_case2;

load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case2
nb_1(1,5)=acc_nb_case2;
lr_1(1,5)=acc_lr_case2;
knn_1(1,5)=acc_knn_case2;
ann_1(1,5)=acc_ann_case2;
elm_1(1,5)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case2_1
lssvm_1(1,5)=acc_lssvm_case2;
sae_1(1,5)=acc_sae_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case2_lpp
lpp_nb_1(1,5)=acc_nb_case2;
lpp_lr_1(1,5)=acc_lr_case2;
lpp_knn_1(1,5)=acc_knn_case2;
lpp_ann_1(1,5)=acc_ann_case2;
lpp_elm_1(1,5)=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case2_1_lpp
lpp_lssvm_1(1,5)=acc_lssvm_case2;
load F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case2_h_elm
h_elm_1(1,5)=acc_h_elm_case2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\case2 nb_1 lr_1 knn_1 ann_1 elm_1 lssvm_1 sae_1 lpp_nb_1 lpp_lr_1 lpp_knn_1 lpp_ann_1 lpp_elm_1 lpp_lssvm_1 h_elm_1

%%case1
subplot(1,2,1);
plot(nb,'o-b');
title('(a) Case 1','FontWeight','bold');
set(gca,'XTick',1:1:7);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'2','3','4','5','6','7','8'});
xlabel('Number of subjects','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
plot(lr,'^-m');
hold on;
plot(knn,'s-g');
hold on;
plot(ann,'+-y');
hold on;
plot(elm,'>-c');
hold on;
plot(lssvm,'p-r');
hold on;
plot(sae,'d-k');

hold on;
plot(lpp_nb,'o--b');
hold on;
plot(lpp_lr,'^--m');
hold on;
plot(lpp_knn,'s--g');
hold on;
plot(lpp_ann,'+--y');
hold on;
plot(lpp_elm,'>--c');
hold on;
plot(lpp_lssvm,'p--r');
hold on;
plot(h_elm,'d--k');
legend('NB','LR','KNN','ANN','ELM','LSSVM','SAE','LPP-NB','LPP-LR','LPP-KNN','LPP-ANN','LPP-ELM','LPP-LSSVM','H-ELM');

%case2
subplot(1,2,2);
plot(nb_1,'o-b');
title('(b) Case 2','FontWeight','bold');
set(gca,'XTick',1:1:5);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'2','3','4','5','6'});
xlabel('Number of subjects','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
grid on;
hold on;
plot(lr_1,'^-m');
hold on;
plot(knn_1,'s-g');
hold on;
plot(ann_1,'+-y');
hold on;
plot(elm_1,'>-c');
hold on;
plot(lssvm_1,'p-r');
hold on;
plot(sae_1,'d-k');

hold on;
plot(lpp_nb_1,'o--b');
hold on;
plot(lpp_lr_1,'^--m');
hold on;
plot(lpp_knn_1,'s--g');
hold on;
plot(lpp_ann_1,'+--y');
hold on;
plot(lpp_elm_1,'>--c');
hold on;
plot(lpp_lssvm_1,'p--r');
hold on;
plot(h_elm_1,'d--k');
