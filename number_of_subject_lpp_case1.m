%NB LR KNN ANN ELM LSSVM LPP-NB LPP-LR LPP-KNN LPP-ANN LPP-ELM LPP-LSSVM
%H-ELM SAE
%任务一，8位被试A,B,C,D,E,F,G,H   按顺序被试增加  k折交叉验证(8>=k>=2)
%average_session

clc;
clear;
close all;
warning off;

%k=2
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
x11=[x1;x2];

rr=70;
%进行LPP
fea =x11;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x11=fea*eigvector;

load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
x22=[x1;x2];
%进行LPP
fea =x22;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x22=fea*eigvector;

x1=x11;
x2=x22;
y=[y;y];
K=2;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%nb
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_1,ytr_1); 
    yte_p=predict(nb,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_2,ytr_2); 
    yte_p=predict(nb,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session2=mean(TestingAccuracy);
end
end
acc_nb_case1=(acc_nb_2_session1+acc_nb_2_session2)./2;

%lr
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    for z=1:1
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:1
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(1800,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(1800,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(1800,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_1,'logit'));
  yte_p_2=round(glmval(model_2,xte_1,'logit'));
  yte_p_3=round(glmval(model_3,xte_1,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session1=mean(TestingAccuracy);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    for z=1:1
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:1
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(1800,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(1800,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(1800,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_2,'logit'));
  yte_p_2=round(glmval(model_2,xte_2,'logit'));
  yte_p_3=round(glmval(model_3,xte_2,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session2=mean(TestingAccuracy);
end
end
acc_lr_case1=(acc_lr_2_session1+acc_lr_2_session2)./2;

%knn
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_1,ytr_1,'NumNeighbors',39);
    yte_p=predict(mdl,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_2,ytr_2,'NumNeighbors',39);
    yte_p=predict(mdl,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session2=mean(TestingAccuracy);
end
end
acc_knn_case1=(acc_knn_2_session1+acc_knn_2_session2)./2;

%ann
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_1',ind2vec((ytr_1)')); 
    yte_p=net(xte_1');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_2',ind2vec((ytr_2)'));
    yte_p=net(xte_2');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session2=mean(TestingAccuracy);
end
end
acc_ann_case1=(acc_ann_2_session1+acc_ann_2_session2)./2;

%elm
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy);
end
end
acc_elm_case1=(acc_elm_2_session1+acc_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case1_lpp acc_nb_case1 acc_lr_case1 acc_knn_case1 acc_ann_case1 acc_elm_case1
    
%k=3
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
x11=[x1;x2;x3];
rr=70;
%进行LPP
fea =x11;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x11=fea*eigvector;
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
x22=[x1;x2;x3];
%进行LPP
fea =x22;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x22=fea*eigvector;
x1=x11;
x2=x22;
y=[y;y;y];
K=3;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%nb
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_1,ytr_1); 
    yte_p=predict(nb,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_2,ytr_2); 
    yte_p=predict(nb,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session2=mean(TestingAccuracy);
end
end
acc_nb_case1=(acc_nb_2_session1+acc_nb_2_session2)./2;

%lr
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    for z=1:2
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:2
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(3600,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(3600,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(3600,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_1,'logit'));
  yte_p_2=round(glmval(model_2,xte_1,'logit'));
  yte_p_3=round(glmval(model_3,xte_1,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session1=mean(TestingAccuracy);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    for z=1:2
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:2
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(3600,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(3600,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(3600,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_2,'logit'));
  yte_p_2=round(glmval(model_2,xte_2,'logit'));
  yte_p_3=round(glmval(model_3,xte_2,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session2=mean(TestingAccuracy);
end
end
acc_lr_case1=(acc_lr_2_session1+acc_lr_2_session2)./2;

%knn
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_1,ytr_1,'NumNeighbors',39);
    yte_p=predict(mdl,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_2,ytr_2,'NumNeighbors',39);
    yte_p=predict(mdl,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session2=mean(TestingAccuracy);
end
end
acc_knn_case1=(acc_knn_2_session1+acc_knn_2_session2)./2;

%ann
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_1',ind2vec((ytr_1)')); 
    yte_p=net(xte_1');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_2',ind2vec((ytr_2)'));
    yte_p=net(xte_2');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session2=mean(TestingAccuracy);
end
end
acc_ann_case1=(acc_ann_2_session1+acc_ann_2_session2)./2;

%elm
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy);
end
end
acc_elm_case1=(acc_elm_2_session1+acc_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case1_lpp acc_nb_case1 acc_lr_case1 acc_knn_case1 acc_ann_case1 acc_elm_case1

%k=4
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
x4=x;
x11=[x1;x2;x3;x4];
rr=70;
%进行LPP
fea =x11;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x11=fea*eigvector;
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
x4=x;
x22=[x1;x2;x3;x4];
%进行LPP
fea =x22;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x22=fea*eigvector;
x1=x11;
x2=x22;
y=[y;y;y;y];
K=4;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%nb
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_1,ytr_1); 
    yte_p=predict(nb,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_2,ytr_2); 
    yte_p=predict(nb,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session2=mean(TestingAccuracy);
end
end
acc_nb_case1=(acc_nb_2_session1+acc_nb_2_session2)./2;

%lr
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    for z=1:3
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:3
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(5400,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(5400,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(5400,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_1,'logit'));
  yte_p_2=round(glmval(model_2,xte_1,'logit'));
  yte_p_3=round(glmval(model_3,xte_1,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session1=mean(TestingAccuracy);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    for z=1:3
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:3
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(5400,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(5400,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(5400,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_2,'logit'));
  yte_p_2=round(glmval(model_2,xte_2,'logit'));
  yte_p_3=round(glmval(model_3,xte_2,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session2=mean(TestingAccuracy);
end
end
acc_lr_case1=(acc_lr_2_session1+acc_lr_2_session2)./2;

%knn
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_1,ytr_1,'NumNeighbors',39);
    yte_p=predict(mdl,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_2,ytr_2,'NumNeighbors',39);
    yte_p=predict(mdl,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session2=mean(TestingAccuracy);
end
end
acc_knn_case1=(acc_knn_2_session1+acc_knn_2_session2)./2;

%ann
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_1',ind2vec((ytr_1)')); 
    yte_p=net(xte_1');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_2',ind2vec((ytr_2)'));
    yte_p=net(xte_2');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session2=mean(TestingAccuracy);
end
end
acc_ann_case1=(acc_ann_2_session1+acc_ann_2_session2)./2;

%elm
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy);
end
end
acc_elm_case1=(acc_elm_2_session1+acc_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case1_lpp acc_nb_case1 acc_lr_case1 acc_knn_case1 acc_ann_case1 acc_elm_case1


%k=5
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
x5=x;
x11=[x1;x2;x3;x4;x5];
rr=70;
%进行LPP
fea =x11;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x11=fea*eigvector;
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_2
x5=x;
x22=[x1;x2;x3;x4;x5];
%进行LPP
fea =x22;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x22=fea*eigvector;
x1=x11;
x2=x22;
y=[y;y;y;y;y];
K=5;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%nb
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_1,ytr_1); 
    yte_p=predict(nb,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_2,ytr_2); 
    yte_p=predict(nb,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session2=mean(TestingAccuracy);
end
end
acc_nb_case1=(acc_nb_2_session1+acc_nb_2_session2)./2;

%lr
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    for z=1:4
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:4
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(7200,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(7200,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(7200,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_1,'logit'));
  yte_p_2=round(glmval(model_2,xte_1,'logit'));
  yte_p_3=round(glmval(model_3,xte_1,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session1=mean(TestingAccuracy);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    for z=1:4
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:4
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(7200,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(7200,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(7200,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_2,'logit'));
  yte_p_2=round(glmval(model_2,xte_2,'logit'));
  yte_p_3=round(glmval(model_3,xte_2,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session2=mean(TestingAccuracy);
end
end
acc_lr_case1=(acc_lr_2_session1+acc_lr_2_session2)./2;

%knn
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_1,ytr_1,'NumNeighbors',39);
    yte_p=predict(mdl,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_2,ytr_2,'NumNeighbors',39);
    yte_p=predict(mdl,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session2=mean(TestingAccuracy);
end
end
acc_knn_case1=(acc_knn_2_session1+acc_knn_2_session2)./2;

%ann
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_1',ind2vec((ytr_1)')); 
    yte_p=net(xte_1');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_2',ind2vec((ytr_2)'));
    yte_p=net(xte_2');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session2=mean(TestingAccuracy);
end
end
acc_ann_case1=(acc_ann_2_session1+acc_ann_2_session2)./2;

%elm
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy);
end
end
acc_elm_case1=(acc_elm_2_session1+acc_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case1_lpp acc_nb_case1 acc_lr_case1 acc_knn_case1 acc_ann_case1 acc_elm_case1


%k=6
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_1
x6=x;
x11=[x1;x2;x3;x4;x5;x6];
rr=70;
%进行LPP
fea =x11;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x11=fea*eigvector;
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_2
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_2
x6=x;
x22=[x1;x2;x3;x4;x5;x6];
%进行LPP
fea =x22;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x22=fea*eigvector;
x1=x11;
x2=x22;
y=[y;y;y;y;y;y];
K=6;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%nb
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_1,ytr_1); 
    yte_p=predict(nb,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_2,ytr_2); 
    yte_p=predict(nb,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session2=mean(TestingAccuracy);
end
end
acc_nb_case1=(acc_nb_2_session1+acc_nb_2_session2)./2;

%lr
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    for z=1:5
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:5
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(9000,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(9000,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(9000,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_1,'logit'));
  yte_p_2=round(glmval(model_2,xte_1,'logit'));
  yte_p_3=round(glmval(model_3,xte_1,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session1=mean(TestingAccuracy);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    for z=1:5
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:5
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(9000,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(9000,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(9000,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_2,'logit'));
  yte_p_2=round(glmval(model_2,xte_2,'logit'));
  yte_p_3=round(glmval(model_3,xte_2,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session2=mean(TestingAccuracy);
end
end
acc_lr_case1=(acc_lr_2_session1+acc_lr_2_session2)./2;

%knn
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_1,ytr_1,'NumNeighbors',39);
    yte_p=predict(mdl,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_2,ytr_2,'NumNeighbors',39);
    yte_p=predict(mdl,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session2=mean(TestingAccuracy);
end
end
acc_knn_case1=(acc_knn_2_session1+acc_knn_2_session2)./2;

%ann
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_1',ind2vec((ytr_1)')); 
    yte_p=net(xte_1');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_2',ind2vec((ytr_2)'));
    yte_p=net(xte_2');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session2=mean(TestingAccuracy);
end
end
acc_ann_case1=(acc_ann_2_session1+acc_ann_2_session2)./2;

%elm
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy);
end
end
acc_elm_case1=(acc_elm_2_session1+acc_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case1_lpp acc_nb_case1 acc_lr_case1 acc_knn_case1 acc_ann_case1 acc_elm_case1


%k=7
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_1
x6=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s7_1
x7=x;
x11=[x1;x2;x3;x4;x5;x6;x7];
rr=70;
%进行LPP
fea =x11;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x11=fea*eigvector;
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_2
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_2
x6=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s7_2
x7=x;
x22=[x1;x2;x3;x4;x5;x6;x7];
%进行LPP
fea =x22;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x22=fea*eigvector;
x1=x11;
x2=x22;
y=[y;y;y;y;y;y;y];
K=7;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%nb
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_1,ytr_1); 
    yte_p=predict(nb,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_2,ytr_2); 
    yte_p=predict(nb,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session2=mean(TestingAccuracy);
end
end
acc_nb_case1=(acc_nb_2_session1+acc_nb_2_session2)./2;

%lr
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    for z=1:6
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:6
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(10800,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(10800,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(10800,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_1,'logit'));
  yte_p_2=round(glmval(model_2,xte_1,'logit'));
  yte_p_3=round(glmval(model_3,xte_1,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session1=mean(TestingAccuracy);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    for z=1:6
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:6
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(10800,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(10800,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(10800,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_2,'logit'));
  yte_p_2=round(glmval(model_2,xte_2,'logit'));
  yte_p_3=round(glmval(model_3,xte_2,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session2=mean(TestingAccuracy);
end
end
acc_lr_case1=(acc_lr_2_session1+acc_lr_2_session2)./2;

%knn
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_1,ytr_1,'NumNeighbors',39);
    yte_p=predict(mdl,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_2,ytr_2,'NumNeighbors',39);
    yte_p=predict(mdl,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session2=mean(TestingAccuracy);
end
end
acc_knn_case1=(acc_knn_2_session1+acc_knn_2_session2)./2;

%ann
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_1',ind2vec((ytr_1)')); 
    yte_p=net(xte_1');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_2',ind2vec((ytr_2)'));
    yte_p=net(xte_2');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session2=mean(TestingAccuracy);
end
end
acc_ann_case1=(acc_ann_2_session1+acc_ann_2_session2)./2;

%elm
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy);
end
end
acc_elm_case1=(acc_elm_2_session1+acc_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject7_case1_lpp acc_nb_case1 acc_lr_case1 acc_knn_case1 acc_ann_case1 acc_elm_case1


%k=8
load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_1
x6=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s7_1
x7=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s8_1
x8=x;
x11=[x1;x2;x3;x4;x5;x6;x7;x8];
rr=70;
%进行LPP
fea =x11;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x11=fea*eigvector;
load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
x1=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
x2=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
x3=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
x4=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s5_2
x5=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s6_2
x6=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s7_2
x7=x;
load F:\matlab\trial_procedure\study_1\features\ex1\s8_2
x8=x;
x22=[x1;x2;x3;x4;x5;x6;x7;x8];
%进行LPP
fea =x22;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x22=fea*eigvector;
x1=x11;
x2=x22;
y=[y;y;y;y;y;y;y;y];
K=8;
[xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
[ytr_all_1, yte_all_1] = kfcv(y,K,'off');
[xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
[ytr_all_2, yte_all_2] = kfcv(y,K,'off');

%nb
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_1,ytr_1); 
    yte_p=predict(nb,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    nb=NaiveBayes.fit(xtr_2,ytr_2); 
    yte_p=predict(nb,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_nb_2_session2=mean(TestingAccuracy);
end
end
acc_nb_case1=(acc_nb_2_session1+acc_nb_2_session2)./2;

%lr
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    for z=1:7
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:7
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_1([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(12600,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(12600,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(12600,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_1,'logit'));
  yte_p_2=round(glmval(model_2,xte_1,'logit'));
  yte_p_3=round(glmval(model_3,xte_1,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session1=mean(TestingAccuracy);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    for z=1:7
    y_low(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700]);
    y_medium(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700]);
    y_high(1+(z-1)*900:900+(z-1)*900,:)=ytr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700]);
    end

    for z=1:7
    x_low(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([1+(z-1)*2700:450+(z-1)*2700 2251+(z-1)*2700:2700+(z-1)*2700],:);
    x_medium(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([451+(z-1)*2700:900+(z-1)*2700 1801+(z-1)*2700:2250+(z-1)*2700],:);
    x_high(1+(z-1)*900:900+(z-1)*900,:)= xtr_2([901+(z-1)*2700:1350+(z-1)*2700 1351+(z-1)*2700:1800+(z-1)*2700],:);
    end
    
%前者“+”，后者“-”
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  model_1=glmfit(x_1,[y_1 ones(12600,1)],'binomial','link','logit');
  model_2=glmfit(x_2,[y_2 ones(12600,1)],'binomial','link','logit');
  model_3=glmfit(x_3,[y_3 ones(12600,1)],'binomial','link','logit');
  yte_p_1=round(glmval(model_1,xte_2,'logit'));
  yte_p_2=round(glmval(model_2,xte_2,'logit'));
  yte_p_3=round(glmval(model_3,xte_2,'logit'));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %目标类别确定之后的混淆矩阵
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lr_2_session2=mean(TestingAccuracy);
end
end
acc_lr_case1=(acc_lr_2_session1+acc_lr_2_session2)./2;

%knn
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_1,ytr_1,'NumNeighbors',39);
    yte_p=predict(mdl,xte_1);
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    mdl=fitcknn(xtr_2,ytr_2,'NumNeighbors',39);
    yte_p=predict(mdl,xte_2);
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_knn_2_session2=mean(TestingAccuracy);
end
end
acc_knn_case1=(acc_knn_2_session1+acc_knn_2_session2)./2;

%ann
%%%%%session1
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_1',ind2vec((ytr_1)')); 
    yte_p=net(xte_1');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    net=patternnet(120);
    net=train(net,xtr_2',ind2vec((ytr_2)'));
    yte_p=net(xte_2');
    yte_p=vec2ind(yte_p);
    yte_p=yte_p';
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%计算混淆矩阵
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %混淆矩阵指标
    TestingAccuracy(i,j)=Acc(i,j);
    acc_ann_2_session2=mean(TestingAccuracy);
end
end
acc_ann_case1=(acc_ann_2_session1+acc_ann_2_session2)./2;

%elm
for j=1
for i=1:K
    xtr_1=cell2mat(xtr_all_1(i,:));
    xte_1=cell2mat(xte_all_1(i,:));
    ytr_1=cell2mat(ytr_all_1(i,:));
    yte_1=cell2mat(yte_all_1(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_1,ytr_1, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1   elm_train_m()最后一个参数就是激活函数参数
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_1,yte_1,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session1=mean(TestingAccuracy);
end
end

%%%session2
for j=1
for i=1:K
    xtr_2=cell2mat(xtr_all_2(i,:));
    xte_2=cell2mat(xte_all_2(i,:));
    ytr_2=cell2mat(ytr_all_2(i,:));
    yte_2=cell2mat(yte_all_2(i,:));
    rand('state',0)
    [TrainingTime(i,j),TrainingAccuracy(i,j),elm_model,label_index_expected] = elm_train_m(xtr_2,ytr_2, 1, 390, 'sig', 2^0);%ELM只认1，2类标签，不认0，1
    [TestingTime(i,j), TestingAccuracy(i,j), ty] = elm_predict(xte_2,yte_2,elm_model);  % TY: the actual output of the testing data
    acc_elm_2_session2=mean(TestingAccuracy);
end
end
acc_elm_case1=(acc_elm_2_session1+acc_elm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject8_case1_lpp acc_nb_case1 acc_lr_case1 acc_knn_case1 acc_ann_case1 acc_elm_case1
