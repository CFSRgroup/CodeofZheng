clc;
clear;
close all;
warning off;

%case1数据
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject8_case1
nb=acc_nb_case1;
lr=acc_lr_case1;
knn=acc_knn_case1;
ann=acc_ann_case1;
elm=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject8_case1_1
lssvm=acc_lssvm_case1;
sae=acc_sae_case1;
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject8_case1_lpp
lpp_nb=acc_nb_case1;
lpp_lr=acc_lr_case1;
lpp_knn=acc_knn_case1;
lpp_ann=acc_ann_case1;
lpp_elm=acc_elm_case1;
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject8_case1_1_lpp
lpp_lssvm=acc_lssvm_case1;
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject8_case1_h_elm
h_elm=acc_h_elm_case1;
save F:\matlab\trial_procedure\study_1\fig13_feature_number\case1 nb lr knn ann elm lssvm sae lpp_nb lpp_lr lpp_knn lpp_ann lpp_elm lpp_lssvm h_elm

%case2数据
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject6_case2
nb_1=acc_nb_case2;
lr_1=acc_lr_case2;
knn_1=acc_knn_case2;
ann_1=acc_ann_case2;
elm_1=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject6_case2_1
lssvm_1=acc_lssvm_case2;
sae_1=acc_sae_case2;
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject6_case2_lpp
lpp_nb_1=acc_nb_case2;
lpp_lr_1=acc_lr_case2;
lpp_knn_1=acc_knn_case2;
lpp_ann_1=acc_ann_case2;
lpp_elm_1=acc_elm_case2;
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject6_case2_1_lpp
lpp_lssvm_1=acc_lssvm_case2;
load F:\matlab\trial_procedure\study_1\fig13_feature_number\subject6_case2_h_elm
h_elm_1=acc_h_elm_case2;
save F:\matlab\trial_procedure\study_1\fig13_feature_number\case2 nb_1 lr_1 knn_1 ann_1 elm_1 lssvm_1 sae_1 lpp_nb_1 lpp_lr_1 lpp_knn_1 lpp_ann_1 lpp_elm_1 lpp_lssvm_1 h_elm_1

%%case1
subplot(1,2,1);
plot(nb,'o-b');
title('(a) Case 1','FontWeight','bold');
set(gca,'XTick',1:17:170);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'0','17','34','51','68','85','102','119','136','153','170'});
xlabel('Feature Number','FontWeight','bold');
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
set(gca,'XTick',1:17:170);   %图形编辑中，XLim改为0:9
set(gca,'XTickLabel',{'0','17','34','51','68','85','102','119','136','153','170'});
xlabel('Feature Number','FontWeight','bold');
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
