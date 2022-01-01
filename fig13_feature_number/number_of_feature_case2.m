clc;
clear;
close all;
warning off;

%case2
%k=6
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_1
x4_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s5_1
x5_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s6_1
x6_1=x;
x33=[x1_1;x2_1;x3_1;x4_1;x5_1;x6_1];
load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_2
x4_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s5_2
x5_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s6_2
x6_1=x;
x44=[x1_1;x2_1;x3_1;x4_1;x5_1;x6_1];
x3=x33;
x4=x44;
y_1=[y;y;y;y;y;y];
K=6;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%nb
%%%%%session1
for j=1:170
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
    nb_1=NaiveBayes.fit(xtr_3(:,1:j),ytr_3);
    yte_p_1=predict(nb_1,xte_3(:,1:j));
    cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵
    [acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy_1(i,j)=Acc_1(i,j);
    acc_nb_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1:170
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
    nb_1=NaiveBayes.fit(xtr_4(:,1:j),ytr_4); 
    yte_p_1=predict(nb_1,xte_4(:,1:j));
    cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
    [acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy_1(i,j)=Acc_1(i,j);
    acc_nb_2_session2=mean(TestingAccuracy_1);
end
end
acc_nb_case2=(acc_nb_2_session1+acc_nb_2_session2)./2;

%lr
%%%%%session1
for j=1:170
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    for z=1:5
    y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_3([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
    y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
    y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:5
    x_low_1(1+(z-1)*580:580+(z-1)*580,:)=xtr_3([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450],:);
    x_medium_1(1+(z-1)*435:435+(z-1)*435,:)=xtr_3([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450],:);
    x_high_1(1+(z-1)*435:435+(z-1)*435,:)=xtr_3([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450],:);
    end
    
%前者“+”，后者“-”
x_4=[x_low_1;x_medium_1];
y_4=[0*ones(length(y_low_1),1);1*ones(length(y_medium_1),1)];

x_5=[x_low_1;x_high_1];
y_5=[0*ones(length(y_low_1),1);1*ones(length(y_high_1),1)];

x_6=[x_medium_1;x_high_1];
y_6=[0*ones(length(y_medium_1),1);1*ones(length(y_high_1),1)];
    rand('state',0)
model_4=glmfit(x_4(:,1:j),[y_4 ones(5075,1)],'binomial','link','logit');
model_5=glmfit(x_5(:,1:j),[y_5 ones(5075,1)],'binomial','link','logit');
model_6=glmfit(x_6(:,1:j),[y_6 ones(4350,1)],'binomial','link','logit');
yte_p_4=round(glmval(model_4,xte_3(:,1:j),'logit'));
yte_p_5=round(glmval(model_5,xte_3(:,1:j),'logit'));
yte_p_6=round(glmval(model_6,xte_3(:,1:j),'logit'));
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lr_2_session1=mean(TestingAccuracy_11);
end
end

%%%%%session2
for j=1:170
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    for z=1:5
    y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_4([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
    y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
    y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:5
    x_low_1(1+(z-1)*580:580+(z-1)*580,:)=xtr_4([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450],:);
    x_medium_1(1+(z-1)*435:435+(z-1)*435,:)=xtr_4([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450],:);
    x_high_1(1+(z-1)*435:435+(z-1)*435,:)=xtr_4([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450],:);
    end
    
%前者“+”，后者“-”
x_4=[x_low_1;x_medium_1];
y_4=[0*ones(length(y_low_1),1);1*ones(length(y_medium_1),1)];

x_5=[x_low_1;x_high_1];
y_5=[0*ones(length(y_low_1),1);1*ones(length(y_high_1),1)];

x_6=[x_medium_1;x_high_1];
y_6=[0*ones(length(y_medium_1),1);1*ones(length(y_high_1),1)];
    rand('state',0)
model_4=glmfit(x_4(:,1:j),[y_4 ones(5075,1)],'binomial','link','logit');
model_5=glmfit(x_5(:,1:j),[y_5 ones(5075,1)],'binomial','link','logit');
model_6=glmfit(x_6(:,1:j),[y_6 ones(4350,1)],'binomial','link','logit');
yte_p_4=round(glmval(model_4,xte_4(:,1:j),'logit'));
yte_p_5=round(glmval(model_5,xte_4(:,1:j),'logit'));
yte_p_6=round(glmval(model_6,xte_4(:,1:j),'logit'));
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lr_2_session2=mean(TestingAccuracy_11);
end
end
acc_lr_case2=(acc_lr_2_session1+acc_lr_2_session2)./2;

%knn
%%%%%session1
for j=1:170
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
mdl_1=fitcknn(xtr_3(:,1:j),ytr_3,'NumNeighbors',50);
yte_p_1=predict(mdl_1,xte_3(:,1:j));
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy_1(i,j)=Acc_1(i,j);
    acc_knn_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1:170
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
mdl_1=fitcknn(xtr_4(:,1:j),ytr_4,'NumNeighbors',50);
yte_p_1=predict(mdl_1,xte_4(:,1:j));
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_knn_2_session2=mean(TestingAccuracy_1);
end
end
acc_knn_case2=(acc_knn_2_session1+acc_knn_2_session2)./2;

%ann
%%%%%session1
for j=1:170
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
net=patternnet(430);
net=train(net,xtr_3(:,1:j)',ind2vec((ytr_3)'));  %变成矩阵,ind2vec()索引到向量,出现的位置为1,其它位置全为0
yte_p_1=net(xte_3(:,1:j)');
yte_p_1=vec2ind(yte_p_1);
yte_p_1=yte_p_1';
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
    acc_ann_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1:170
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
net=patternnet(430);
net=train(net,xtr_4(:,1:j)',ind2vec((ytr_4)'));  %变成矩阵,ind2vec()索引到向量,出现的位置为1,其它位置全为0
yte_p_1=net(xte_4(:,1:j)');
yte_p_1=vec2ind(yte_p_1);
yte_p_1=yte_p_1';
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
    acc_ann_2_session2=mean(TestingAccuracy_1);
end
end
acc_ann_case2=(acc_ann_2_session1+acc_ann_2_session2)./2;

%elm
for j=1:170
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    rand('state',0)
    [TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_3(:,1:j),ytr_3, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_3(:,1:j),yte_3,elm_model_1);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1:170
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    rand('state',0)
    [TrainingTime_1(i,j),TrainingAccuracy_1(i,j),elm_model_1,label_index_expected_1] = elm_train_m(xtr_4(:,1:j),ytr_4, 1, 60, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
[TestingTime_1(i,j), TestingAccuracy_1(i,j), ty_1] = elm_predict(xte_4(:,1:j),yte_4,elm_model_1);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy_1);
end
end
acc_elm_case2=(acc_elm_2_session1+acc_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\fig13_feature_number\subject6_case2 acc_nb_case2 acc_lr_case2 acc_knn_case2 acc_ann_case2 acc_elm_case2
