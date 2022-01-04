clc;
clear;
close all;
warning off;

% %case1_session1
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session1\BC_test
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session1\BC_train

% %case1_session2
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session2\BC_test_2
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session2\BC_train_2

% %case2_session1
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session1\BC_test_3
% load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session1\BC_train_3
% 
%case2_session2
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session2\BC_test_4
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session2\BC_train_4

% 基分类器SDAE训练和测试，任务一：8x7
% %被试1做测试集
% for i=1:7
%     rand('state',0)
%     sae = saesetup([70 10 85]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC1_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 10 85 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC1_' num2str(i) ',y_train_BC1_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels1_' num2str(i) '  =nnpredict(nn,BC_test1); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member1=[labels1_1 labels1_2 labels1_3 labels1_4 labels1_5 labels1_6 labels1_7];
% Time1=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member1 Ensemble_member1 Time1

% %被试2做测试集
% for i=1:7
%     rand('state',0)
%     sae = saesetup([70 10 85]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC2_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 10 85 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC2_' num2str(i) ',y_train_BC2_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels2_' num2str(i) '  =nnpredict(nn,BC_test2); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member2=[labels2_1 labels2_2 labels2_3 labels2_4 labels2_5 labels2_6 labels2_7];
% Time2=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member2 Ensemble_member2 Time2

% %被试3做测试集
% for i=1:7
%     rand('state',0)
%     sae = saesetup([70 10 85]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC3_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 10 85 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC3_' num2str(i) ',y_train_BC3_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels3_' num2str(i) '  =nnpredict(nn,BC_test3); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member3=[labels3_1 labels3_2 labels3_3 labels3_4 labels3_5 labels3_6 labels3_7];
% Time3=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member3 Ensemble_member3 Time3

% %被试4做测试集
% for i=1:7
%     rand('state',0)
%     sae = saesetup([70 10 85]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC4_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 10 85 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC4_' num2str(i) ',y_train_BC4_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels4_' num2str(i) '  =nnpredict(nn,BC_test4); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member4=[labels4_1 labels4_2 labels4_3 labels4_4 labels4_5 labels4_6 labels4_7];
% Time4=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member4 Ensemble_member4 Time4

% %被试5做测试集
% for i=1:7
%     rand('state',0)
%     sae = saesetup([70 10 85]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC5_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 10 85 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC5_' num2str(i) ',y_train_BC5_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels5_' num2str(i) '  =nnpredict(nn,BC_test5); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member5=[labels5_1 labels5_2 labels5_3 labels5_4 labels5_5 labels5_6 labels5_7];
% Time5=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member5 Ensemble_member5 Time5

% %被试6做测试集
% for i=1:7
%     rand('state',0)
%     sae = saesetup([70 10 85]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC6_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 10 85 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC6_' num2str(i) ',y_train_BC6_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels6_' num2str(i) '  =nnpredict(nn,BC_test6); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member6=[labels6_1 labels6_2 labels6_3 labels6_4 labels6_5 labels6_6 labels6_7];
% Time6=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member6 Ensemble_member6 Time6

% %被试7做测试集
% for i=1:7
%     rand('state',0)
%     sae = saesetup([70 10 85]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC7_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 10 85 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC7_' num2str(i) ',y_train_BC7_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels7_' num2str(i) '  =nnpredict(nn,BC_test7); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member7=[labels7_1 labels7_2 labels7_3 labels7_4 labels7_5 labels7_6 labels7_7];
% Time7=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member7 Ensemble_member7 Time7

% %被试8做测试集
% for i=1:7
%     rand('state',0)
%     sae = saesetup([70 10 85]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC8_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 10 85 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC8_' num2str(i) ',y_train_BC8_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels8_' num2str(i) '  =nnpredict(nn,BC_test8); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member8=[labels8_1 labels8_2 labels8_3 labels8_4 labels8_5 labels8_6 labels8_7];
% Time8=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case1_session2\Ensemble_member8 Ensemble_member8 Time8










% 基分类器SDAE训练和测试，任务二：6x5
% %被试1做测试集
% for i=1:5
%     rand('state',0)
%     sae = saesetup([70 15 5]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC1_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 15 5 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC1_' num2str(i) ',y_train_BC1_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels1_' num2str(i) '  =nnpredict(nn,BC_test1); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member1=[labels1_1 labels1_2 labels1_3 labels1_4 labels1_5];
% Time1=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session2\Ensemble_member1 Ensemble_member1 Time1

% %被试2做测试集
% for i=1:5
%     rand('state',0)
%     sae = saesetup([70 15 5]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC2_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 15 5 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC2_' num2str(i) ',y_train_BC2_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels2_' num2str(i) '  =nnpredict(nn,BC_test2); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member2=[labels2_1 labels2_2 labels2_3 labels2_4 labels2_5];
% Time2=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session2\Ensemble_member2 Ensemble_member2 Time2

% %被试3做测试集
% for i=1:5
%     rand('state',0)
%     sae = saesetup([70 15 5]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC3_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 15 5 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC3_' num2str(i) ',y_train_BC3_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels3_' num2str(i) '  =nnpredict(nn,BC_test3); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member3=[labels3_1 labels3_2 labels3_3 labels3_4 labels3_5];
% Time3=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session2\Ensemble_member3 Ensemble_member3 Time3

% %被试4做测试集
% for i=1:5
%     rand('state',0)
%     sae = saesetup([70 15 5]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC4_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 15 5 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC4_' num2str(i) ',y_train_BC4_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels4_' num2str(i) '  =nnpredict(nn,BC_test4); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member4=[labels4_1 labels4_2 labels4_3 labels4_4 labels4_5];
% Time4=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session2\Ensemble_member4 Ensemble_member4 Time4

% %被试5做测试集
% for i=1:5
%     rand('state',0)
%     sae = saesetup([70 15 5]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC5_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 15 5 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC5_' num2str(i) ',y_train_BC5_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels5_' num2str(i) '  =nnpredict(nn,BC_test5); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member5=[labels5_1 labels5_2 labels5_3 labels5_4 labels5_5];
% Time5=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session2\Ensemble_member5 Ensemble_member5 Time5

% %被试6做测试集
% for i=1:5
%     rand('state',0)
%     sae = saesetup([70 15 5]);
%     sae.ae{1}.activation_function       = 'sigm';   %第一隐层
%     sae.ae{1}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     sae.ae{2}.activation_function       = 'sigm';   %第二隐层
%     sae.ae{2}.learningRate              = 0.01;%定义学习率
%     sae.ae{1}.inputZeroMaskedFraction = 0.2; 
%     opts.numepochs =   20;%重复迭代次数
%     opts.batchsize = 25;%数据块大小--每一块用于一次梯度下降算法
%     start_time_train=cputime;
%     eval(['sae = saetrain(sae, BC6_' num2str(i) ', opts);'])
% % Use the SDAE to initialize a FFNN
% nn = nnsetup([70 15 5 3]);
% nn.activation_function              = 'sigm';
% nn.learningRate                     = 0.01;
% nn.W{1} = sae.ae{1}.W{1};%更新了nn的权值
% nn.W{2} = sae.ae{2}.W{1};
% 
% % Train the FFNN
% opts.numepochs =   20;
% opts.batchsize = 25;
% 
% % start_time_train=cputime;
% eval(['nn = nntrain(nn,BC6_' num2str(i) ',y_train_BC6_' num2str(i) ', opts);']);
% end_time_train=cputime;
% TrainingTime1=end_time_train-start_time_train;
% 
% start_time_test=cputime;
% % [~, ~, labels] = nntest(nn, xte_1, yte_1);%测试
% eval(['labels6_' num2str(i) '  =nnpredict(nn,BC_test6); '])
% end_time_test=cputime;
% TestingTime1=end_time_test-start_time_test; 
% end
% 
% Ensemble_member6=[labels6_1 labels6_2 labels6_3 labels6_4 labels6_5];
% Time6=TrainingTime1+TestingTime1;
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\meta_data\case2_session2\Ensemble_member6 Ensemble_member6 Time6


