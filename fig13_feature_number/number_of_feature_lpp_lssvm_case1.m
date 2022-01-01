clc;
clear;
close all;
warning off;

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
%����LPP
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
%����LPP
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

%lssvm
%%%%%session1
for j=1:70
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
    
%ǰ�ߡ�+�������ߡ�-��
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
  [alpha_1,b_1] = trainlssvm({x_1(:,1:j),y_1,'c',2^-8,[],'lin_kernel'});   %[alpha, b] = trainlssvm({X,Y,type,gam,kernel_par,kernel,preprocess})   gam�ǳͷ�ϵ��C
  [alpha_2,b_2] = trainlssvm({x_2(:,1:j),y_2,'c',2^-8,[],'lin_kernel'});
  [alpha_3,b_3] = trainlssvm({x_3(:,1:j),y_3,'c',2^-8,[],'lin_kernel'});
  yte_p_1=simlssvm({x_1(:,1:j),y_1,'c',2^-8,[],'lin_kernel'},{alpha_1,b_1},xte_1(:,1:j));
  yte_p_2=simlssvm({x_2(:,1:j),y_2,'c',2^-8,[],'lin_kernel'},{alpha_2,b_2},xte_1(:,1:j));
  yte_p_3=simlssvm({x_3(:,1:j),y_3,'c',2^-8,[],'lin_kernel'},{alpha_3,b_3},xte_1(:,1:j));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %Ŀ�����ȷ��֮��Ļ�������
    cte(:,:,i,j)=cfmatrix(yte_1,yte_p);%�����������
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %��������ָ��
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lssvm_2_session1=mean(TestingAccuracy);
end
end

%%%%%session2
for j=1:70
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
    
%ǰ�ߡ�+�������ߡ�-��
x_1=[x_low;x_medium];
y_1=[0*ones(length(y_low),1);1*ones(length(y_medium),1)];

x_2=[x_low;x_high];
y_2=[0*ones(length(y_low),1);1*ones(length(y_high),1)];

x_3=[x_medium;x_high];
y_3=[0*ones(length(y_medium),1);1*ones(length(y_high),1)];
    rand('state',0)
    [alpha_1,b_1] = trainlssvm({x_1(:,1:j),y_1,'c',2^-8,[],'lin_kernel'});
    [alpha_2,b_2] = trainlssvm({x_2(:,1:j),y_2,'c',2^-8,[],'lin_kernel'});
    [alpha_3,b_3] = trainlssvm({x_3(:,1:j),y_3,'c',2^-8,[],'lin_kernel'});
    yte_p_1=simlssvm({x_1(:,1:j),y_1,'c',2^-8,[],'lin_kernel'},{alpha_1,b_1},xte_2(:,1:j));
    yte_p_2=simlssvm({x_2(:,1:j),y_2,'c',2^-8,[],'lin_kernel'},{alpha_2,b_2},xte_2(:,1:j));
    yte_p_3=simlssvm({x_3(:,1:j),y_3,'c',2^-8,[],'lin_kernel'},{alpha_3,b_3},xte_2(:,1:j));
   for z=1:2700
    yte_p(z,:)=ovo_code(yte_p_1(1+(z-1)*1,:),yte_p_2(1+(z-1)*1,:),yte_p_3(1+(z-1)*1,:));
   end
    %Ŀ�����ȷ��֮��Ļ�������
    cte(:,:,i,j)=cfmatrix(yte_2,yte_p);%�����������
    [acc_low(i,j),acc_medium(i,j),acc_high(i,j),Acc(i,j),sen(i,j),spe(i,j),pre(i,j),npv(i,j),f1(i,j),fnr(i,j),fpr(i,j),fdr(i,j),foe(i,j),mcc(i,j),bm(i,j),mk(i,j),ka(i,j)] = per_thrtotwo(cte(:,:,i,j)); %��������ָ��
    TestingAccuracy(i,j)=Acc(i,j);
    acc_lssvm_2_session2=mean(TestingAccuracy);
end
end
acc_lssvm_case1=(acc_lssvm_2_session1+acc_lssvm_2_session2)./2;
save F:\matlab\trial_procedure\study_1\fig13_feature_number\subject8_case1_1_lpp acc_lssvm_case1


