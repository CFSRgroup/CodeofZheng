%LSSVM SAE
%任务二，6位被试I,J,K,L,M,N   按顺序被试增加  k折交叉验证(6>=k>=2)
%average_session

%k=2
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
x33=[x1_1;x2_1];
load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
x44=[x1_1;x2_1];
x3=x33;
x4=x44;
y_1=[y;y];
K=2;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%lssvm
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    for z=1:1
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_3([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:1
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_3);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_3);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_3);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session1=mean(TestingAccuracy_11);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    for z=1:1
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_4([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:1
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_4);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_4);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_4);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session2=mean(TestingAccuracy_11);
end
end
acc_lssvm_case2=(acc_lssvm_2_session1+acc_lssvm_2_session2)./2;

%sae
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   rand('state',0)  %resets the generator to its initial state

%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);
labels = nnpredict(nn, xte_3);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));

   rand('state',0)  %resets the generator to its initial state
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;
nn = nntrain(nn, xtr_4, ytr_4, opts);
labels = nnpredict(nn, xte_4);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session2=mean(TestingAccuracy_1);
end
end
acc_sae_case2=(acc_sae_2_session1+acc_sae_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject2_case2_1 acc_lssvm_case2 acc_sae_case2

%k=3
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
x3_1=x;
x33=[x1_1;x2_1;x3_1];
load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
x3_1=x;
x44=[x1_1;x2_1;x3_1];
x3=x33;
x4=x44;
y_1=[y;y;y];
K=3;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%lssvm
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    for z=1:2
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_3([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:2
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_3);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_3);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_3);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session1=mean(TestingAccuracy_11);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    for z=1:2
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_4([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:2
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_4);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_4);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_4);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session2=mean(TestingAccuracy_11);
end
end
acc_lssvm_case2=(acc_lssvm_2_session1+acc_lssvm_2_session2)./2;

%sae
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   rand('state',0)  %resets the generator to its initial state

%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);
labels = nnpredict(nn, xte_3);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));

   rand('state',0)  %resets the generator to its initial state
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;
nn = nntrain(nn, xtr_4, ytr_4, opts);
labels = nnpredict(nn, xte_4);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session2=mean(TestingAccuracy_1);
end
end
acc_sae_case2=(acc_sae_2_session1+acc_sae_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject3_case2_1 acc_lssvm_case2 acc_sae_case2


%k=4
load F:\matlab\trial_procedure\study_1\features\ex2\s1_1
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_1
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_1
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_1
x4_1=x;
x33=[x1_1;x2_1;x3_1;x4_1];
load F:\matlab\trial_procedure\study_1\features\ex2\s1_2
x1_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s2_2
x2_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s3_2
x3_1=x;
load F:\matlab\trial_procedure\study_1\features\ex2\s4_2
x4_1=x;
x44=[x1_1;x2_1;x3_1;x4_1];
x3=x33;
x4=x44;
y_1=[y;y;y;y];
K=4;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%lssvm
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    for z=1:3
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_3([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:3
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_3);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_3);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_3);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session1=mean(TestingAccuracy_11);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    for z=1:3
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_4([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:3
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_4);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_4);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_4);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session2=mean(TestingAccuracy_11);
end
end
acc_lssvm_case2=(acc_lssvm_2_session1+acc_lssvm_2_session2)./2;

%sae
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   rand('state',0)  %resets the generator to its initial state

%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);
labels = nnpredict(nn, xte_3);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));

   rand('state',0)  %resets the generator to its initial state
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;
nn = nntrain(nn, xtr_4, ytr_4, opts);
labels = nnpredict(nn, xte_4);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session2=mean(TestingAccuracy_1);
end
end
acc_sae_case2=(acc_sae_2_session1+acc_sae_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject4_case2_1 acc_lssvm_case2 acc_sae_case2


%k=5
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
x33=[x1_1;x2_1;x3_1;x4_1;x5_1];
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
x44=[x1_1;x2_1;x3_1;x4_1;x5_1];
x3=x33;
x4=x44;
y_1=[y;y;y;y;y];
K=5;
[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
[ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');
[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
[ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

%lssvm
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    for z=1:4
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_3([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_3([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:4
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_3);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_3);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_3);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session1=mean(TestingAccuracy_11);
end
end

%%%%%session2
for j=1
for i=1:K
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));
    for z=1:4
   y_low_1(1+(z-1)*580:580+(z-1)*580,:)=ytr_4([1+(z-1)*1450:145+(z-1)*1450 436+(z-1)*1450:580+(z-1)*1450 871+(z-1)*1450:1015+(z-1)*1450 1306+(z-1)*1450:1450+(z-1)*1450]);
   y_medium_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([146+(z-1)*1450:290+(z-1)*1450 581+(z-1)*1450:725+(z-1)*1450 1016+(z-1)*1450:1160+(z-1)*1450 ]);
   y_high_1(1+(z-1)*435:435+(z-1)*435,:)=ytr_4([291+(z-1)*1450:435+(z-1)*1450 726+(z-1)*1450:870+(z-1)*1450 1161+(z-1)*1450:1305+(z-1)*1450]);
    end

    for z=1:4
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_4);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_4);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_4);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session2=mean(TestingAccuracy_11);
end
end
acc_lssvm_case2=(acc_lssvm_2_session1+acc_lssvm_2_session2)./2;

%sae
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   rand('state',0)  %resets the generator to its initial state

%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);
labels = nnpredict(nn, xte_3);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));

   rand('state',0)  %resets the generator to its initial state
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;
nn = nntrain(nn, xtr_4, ytr_4, opts);
labels = nnpredict(nn, xte_4);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session2=mean(TestingAccuracy_1);
end
end
acc_sae_case2=(acc_sae_2_session1+acc_sae_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject5_case2_1 acc_lssvm_case2 acc_sae_case2


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

%lssvm
%%%%%session1
for j=1
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_3);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_3);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_3);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session1=mean(TestingAccuracy_11);
end
end

%%%%%session2
for j=1
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
[alpha_4,b_4] = trainlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'});
[alpha_5,b_5] = trainlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'});
[alpha_6,b_6] = trainlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'});
yte_p_4=simlssvm({x_4,y_4,'c',2^-12,[],'lin_kernel'},{alpha_4,b_4},xte_4);
yte_p_5=simlssvm({x_5,y_5,'c',2^-12,[],'lin_kernel'},{alpha_5,b_5},xte_4);
yte_p_6=simlssvm({x_6,y_6,'c',2^-12,[],'lin_kernel'},{alpha_6,b_6},xte_4);
for z=1:1450
yte_p_11(z,:)=ovo_code(yte_p_4(1+(z-1)*1,:),yte_p_5(1+(z-1)*1,:),yte_p_6(1+(z-1)*1,:));
end
    %目标类别确定之后的混淆矩阵
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_11);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_11(i,j)=Acc_1(i,j);
    acc_lssvm_2_session2=mean(TestingAccuracy_11);
end
end
acc_lssvm_case2=(acc_lssvm_2_session1+acc_lssvm_2_session2)./2;

%sae
%%%%%session1
for j=1
for i=1:K
    xtr_3=cell2mat(xtr_all_3(i,:));
    xte_3=cell2mat(xte_all_3(i,:));
    ytr_3=cell2mat(ytr_all_3(i,:));
    yte_3=cell2mat(yte_all_3(i,:));
    
   rand('state',0)  %resets the generator to its initial state

%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_3, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;

nn = nntrain(nn, xtr_3, ytr_3, opts);
labels = nnpredict(nn, xte_3);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_3,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session1=mean(TestingAccuracy_1);
end
end

%%%session2
for j=1
for i=1:K
    
    xtr_4=cell2mat(xtr_all_4(i,:));
    xte_4=cell2mat(xte_all_4(i,:));
    ytr_4=cell2mat(ytr_all_4(i,:));
    yte_4=cell2mat(yte_all_4(i,:));

   rand('state',0)  %resets the generator to its initial state
   
%  Setup and train a stacked denoising autoencoder (SDAE)
sae = saesetup([170 95 40]);
sae.ae{1}.activation_function       = 'sigm';   %第一隐层
sae.ae{1}.learningRate              = 0.01;%定义学习率
% sae.ae{1}.learningRate              = 1;%定义学习率
% sae.ae{1}.inputZeroMaskedFraction   = 0.01;%以一定概率将输入值变成0，增加输入的鲁棒性，抗噪声能力

sae.ae{2}.activation_function       = 'sigm';   %第二隐层
sae.ae{2}.learningRate              = 0.01;%定义学习率
% sae.ae{2}.learningRate              = 1;%定义学习率
% sae.ae{2}.inputZeroMaskedFraction   = 0.01;

opts.numepochs =   20;%重复迭代次数
opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
sae = saetrain(sae, xtr_4, opts);   

% Use the SDAE to initialize a FFNN
nn = nnsetup([170 95 40 3]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 0.01;
nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   20;
opts.batchsize = 25;
nn = nntrain(nn, xtr_4, ytr_4, opts);
labels = nnpredict(nn, xte_4);
yte_p_1=labels;
cte_1(:,:,i,j)=cfmatrix(yte_4,yte_p_1);%计算混淆矩阵
[acc_low_1(i,j),acc_medium_1(i,j),acc_high_1(i,j),Acc_1(i,j),sen_1(i,j),spe_1(i,j),pre_1(i,j),npv_1(i,j),f1_1(i,j),fnr_1(i,j),fpr_1(i,j),fdr_1(i,j),foe_1(i,j),mcc_1(i,j),bm_1(i,j),mk_1(i,j),ka_1(i,j)] = per_thrtotwo(cte_1(:,:,i,j)); %混淆矩阵指标
TestingAccuracy_1(i,j)=Acc_1(i,j);
acc_sae_2_session2=mean(TestingAccuracy_1);
end
end
acc_sae_case2=(acc_sae_2_session1+acc_sae_2_session2)./2;
save F:\matlab\trial_procedure\study_1\number_of_subjects\subject6_case2_1 acc_lssvm_case2 acc_sae_case2


