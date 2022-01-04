%sdae_elm跨被试（任务一8位）
clc;
clear;
close all;
warning off;

% %case1
% load F:\matlab\trial_procedure\study_1\features\ex1\s1_1
% x1=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s2_1
% x2=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s3_1
% x3=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s4_1
% x4=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s5_1
% x5=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s6_1
% x6=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s7_1
% x7=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s8_1
% x8=x;
% 
% x11=[x1;x2;x3;x4;x5;x6;x7;x8];
% 
% rr=70;
% %进行LPP
% fea =x11;
% options = [];
% options.Metric = 'Euclidean';
% % options.NeighborMode = 'Supervised';
% % options.gnd = y;
% options.ReducedDim=rr;
% W = constructW(fea,options);      
% options.PCARatio = 0.8;    %默认1
% [eigvector, eigvalue] = LPP(W, options, fea);
% x11=fea*eigvector;
% 
% load F:\matlab\trial_procedure\study_1\features\ex1\s1_2
% x1=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s2_2
% x2=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s3_2
% x3=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s4_2
% x4=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s5_2
% x5=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s6_2
% x6=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s7_2
% x7=x;
% load F:\matlab\trial_procedure\study_1\features\ex1\s8_2
% x8=x;
% 
% x22=[x1;x2;x3;x4;x5;x6;x7;x8];
% 
% %进行LPP
% fea =x22;
% options = [];
% options.Metric = 'Euclidean';
% % options.NeighborMode = 'Supervised';
% % options.gnd = y;
% options.ReducedDim=rr;
% W = constructW(fea,options);      
% options.PCARatio = 0.8;   %默认1
% [eigvector, eigvalue] = LPP(W, options, fea);
% x22=fea*eigvector;
% 
% %任务一两个阶段的数据集
% x1=x11;
% x2=x22;
% 
% % y=[y;y;y;y;y;y;y;y];
% 
% K=8;
% low=size(find(y==1));
% medium=size(find(y==2));
% high=size(find(y==3));
% 
% [xtr_all_1, xte_all_1] = kfcv(x1,K,'off');
% % [ytr_all_1, yte_all_1] = kfcv(y,K,'off');
% 
% [xtr_all_2, xte_all_2] = kfcv(x2,K,'off');
% % [ytr_all_2, yte_all_2] = kfcv(y,K,'off');

% %case1_session1
% %基分类器数据输入准备
% a1=cell2mat(xtr_all_1(1,:));
% a2=cell2mat(xtr_all_1(2,:));
% a3=cell2mat(xtr_all_1(3,:));
% a4=cell2mat(xtr_all_1(4,:));
% a5=cell2mat(xtr_all_1(5,:));
% a6=cell2mat(xtr_all_1(6,:));
% a7=cell2mat(xtr_all_1(7,:));
% a8=cell2mat(xtr_all_1(8,:));
% 
% %输入数据附标签
% a1(:,71)=repmat(y,7,1);
% a2(:,71)=repmat(y,7,1);
% a3(:,71)=repmat(y,7,1);
% a4(:,71)=repmat(y,7,1);
% a5(:,71)=repmat(y,7,1);
% a6(:,71)=repmat(y,7,1);
% a7(:,71)=repmat(y,7,1);
% a8(:,71)=repmat(y,7,1);
% 
% %剩余7位被试分开
% a1_1=a1(1:2700,:); a1_2=a1(2701:5400,:); a1_3=a1(5401:8100,:); a1_4=a1(8101:10800,:); a1_5=a1(10801:13500,:); a1_6=a1(13501:16200,:); a1_7=a1(16201:18900,:);
% a2_1=a2(1:2700,:); a2_2=a2(2701:5400,:); a2_3=a2(5401:8100,:); a2_4=a2(8101:10800,:); a2_5=a2(10801:13500,:); a2_6=a2(13501:16200,:); a2_7=a2(16201:18900,:);
% a3_1=a3(1:2700,:); a3_2=a3(2701:5400,:); a3_3=a3(5401:8100,:); a3_4=a3(8101:10800,:); a3_5=a3(10801:13500,:); a3_6=a3(13501:16200,:); a3_7=a3(16201:18900,:);
% a4_1=a4(1:2700,:); a4_2=a4(2701:5400,:); a4_3=a4(5401:8100,:); a4_4=a4(8101:10800,:); a4_5=a4(10801:13500,:); a4_6=a4(13501:16200,:); a4_7=a4(16201:18900,:);
% a5_1=a5(1:2700,:); a5_2=a5(2701:5400,:); a5_3=a5(5401:8100,:); a5_4=a5(8101:10800,:); a5_5=a5(10801:13500,:); a5_6=a5(13501:16200,:); a5_7=a5(16201:18900,:);
% a6_1=a6(1:2700,:); a6_2=a6(2701:5400,:); a6_3=a6(5401:8100,:); a6_4=a6(8101:10800,:); a6_5=a6(10801:13500,:); a6_6=a6(13501:16200,:); a6_7=a6(16201:18900,:);
% a7_1=a7(1:2700,:); a7_2=a7(2701:5400,:); a7_3=a7(5401:8100,:); a7_4=a7(8101:10800,:); a7_5=a7(10801:13500,:); a7_6=a7(13501:16200,:); a7_7=a7(16201:18900,:);
% a8_1=a8(1:2700,:); a8_2=a8(2701:5400,:); a8_3=a8(5401:8100,:); a8_4=a8(8101:10800,:); a8_5=a8(10801:13500,:); a8_6=a8(13501:16200,:); a8_7=a8(16201:18900,:);
% 
% %分别取7位被试的训练集
% n=2600; idx=randperm(2700); idx=idx(1:n);
% BC1_1=a1_1(idx,:); BC1_2=a1_2(idx,:); BC1_3=a1_3(idx,:); BC1_4=a1_4(idx,:); BC1_5=a1_5(idx,:); BC1_6=a1_6(idx,:); BC1_7=a1_7(idx,:);
% BC2_1=a2_1(idx,:); BC2_2=a2_2(idx,:); BC2_3=a2_3(idx,:); BC2_4=a2_4(idx,:); BC2_5=a2_5(idx,:); BC2_6=a2_6(idx,:); BC2_7=a2_7(idx,:);
% BC3_1=a3_1(idx,:); BC3_2=a3_2(idx,:); BC3_3=a3_3(idx,:); BC3_4=a3_4(idx,:); BC3_5=a3_5(idx,:); BC3_6=a3_6(idx,:); BC3_7=a3_7(idx,:);
% BC4_1=a4_1(idx,:); BC4_2=a4_2(idx,:); BC4_3=a4_3(idx,:); BC4_4=a4_4(idx,:); BC4_5=a4_5(idx,:); BC4_6=a4_6(idx,:); BC4_7=a4_7(idx,:);
% BC5_1=a5_1(idx,:); BC5_2=a5_2(idx,:); BC5_3=a5_3(idx,:); BC5_4=a5_4(idx,:); BC5_5=a5_5(idx,:); BC5_6=a5_6(idx,:); BC5_7=a5_7(idx,:);
% BC6_1=a6_1(idx,:); BC6_2=a6_2(idx,:); BC6_3=a6_3(idx,:); BC6_4=a6_4(idx,:); BC6_5=a6_5(idx,:); BC6_6=a6_6(idx,:); BC6_7=a6_7(idx,:);
% BC7_1=a7_1(idx,:); BC7_2=a7_2(idx,:); BC7_3=a7_3(idx,:); BC7_4=a7_4(idx,:); BC7_5=a7_5(idx,:); BC7_6=a7_6(idx,:); BC7_7=a7_7(idx,:);
% BC8_1=a8_1(idx,:); BC8_2=a8_2(idx,:); BC8_3=a8_3(idx,:); BC8_4=a8_4(idx,:); BC8_5=a8_5(idx,:); BC8_6=a8_6(idx,:); BC8_7=a8_7(idx,:);
% %7个基分类器训练集标签
% y_train_BC1_1=BC1_1(:,71);y_train_BC1_2=BC1_2(:,71);y_train_BC1_3=BC1_3(:,71);y_train_BC1_4=BC1_4(:,71);y_train_BC1_5=BC1_5(:,71);y_train_BC1_6=BC1_6(:,71);y_train_BC1_7=BC1_7(:,71);
% y_train_BC2_1=BC2_1(:,71);y_train_BC2_2=BC2_2(:,71);y_train_BC2_3=BC2_3(:,71);y_train_BC2_4=BC2_4(:,71);y_train_BC2_5=BC2_5(:,71);y_train_BC2_6=BC2_6(:,71);y_train_BC2_7=BC2_7(:,71);
% y_train_BC3_1=BC3_1(:,71);y_train_BC3_2=BC3_2(:,71);y_train_BC3_3=BC3_3(:,71);y_train_BC3_4=BC3_4(:,71);y_train_BC3_5=BC3_5(:,71);y_train_BC3_6=BC3_6(:,71);y_train_BC3_7=BC3_7(:,71);
% y_train_BC4_1=BC4_1(:,71);y_train_BC4_2=BC4_2(:,71);y_train_BC4_3=BC4_3(:,71);y_train_BC4_4=BC4_4(:,71);y_train_BC4_5=BC4_5(:,71);y_train_BC4_6=BC4_6(:,71);y_train_BC4_7=BC4_7(:,71);
% y_train_BC5_1=BC5_1(:,71);y_train_BC5_2=BC5_1(:,71);y_train_BC5_3=BC5_3(:,71);y_train_BC5_4=BC5_4(:,71);y_train_BC5_5=BC5_5(:,71);y_train_BC5_6=BC5_6(:,71);y_train_BC5_7=BC5_7(:,71);
% y_train_BC6_1=BC6_1(:,71);y_train_BC6_2=BC6_2(:,71);y_train_BC6_3=BC6_3(:,71);y_train_BC6_4=BC6_4(:,71);y_train_BC6_5=BC6_5(:,71);y_train_BC6_6=BC6_6(:,71);y_train_BC6_7=BC6_7(:,71);
% y_train_BC7_1=BC7_1(:,71);y_train_BC7_2=BC7_2(:,71);y_train_BC7_3=BC7_3(:,71);y_train_BC7_4=BC7_4(:,71);y_train_BC7_5=BC7_5(:,71);y_train_BC7_6=BC7_6(:,71);y_train_BC7_7=BC7_7(:,71);
% y_train_BC8_1=BC8_1(:,71);y_train_BC8_2=BC8_2(:,71);y_train_BC8_3=BC8_3(:,71);y_train_BC8_4=BC8_4(:,71);y_train_BC8_5=BC8_5(:,71);y_train_BC8_6=BC8_6(:,71);y_train_BC8_7=BC8_7(:,71);
% 
% %%%%
% %7个基分类器训练集
% BC1_1(:,71)=[]; BC1_2(:,71)=[]; BC1_3(:,71)=[]; BC1_4(:,71)=[]; BC1_5(:,71)=[]; BC1_6(:,71)=[]; BC1_7(:,71)=[];
% BC2_1(:,71)=[]; BC2_2(:,71)=[]; BC2_3(:,71)=[]; BC2_4(:,71)=[]; BC2_5(:,71)=[]; BC2_6(:,71)=[]; BC2_7(:,71)=[];
% BC3_1(:,71)=[]; BC3_2(:,71)=[]; BC3_3(:,71)=[]; BC3_4(:,71)=[]; BC3_5(:,71)=[]; BC3_6(:,71)=[]; BC3_7(:,71)=[];
% BC4_1(:,71)=[]; BC4_2(:,71)=[]; BC4_3(:,71)=[]; BC4_4(:,71)=[]; BC4_5(:,71)=[]; BC4_6(:,71)=[]; BC4_7(:,71)=[];
% BC5_1(:,71)=[]; BC5_2(:,71)=[]; BC5_3(:,71)=[]; BC5_4(:,71)=[]; BC5_5(:,71)=[]; BC5_6(:,71)=[]; BC5_7(:,71)=[];
% BC6_1(:,71)=[]; BC6_2(:,71)=[]; BC6_3(:,71)=[]; BC6_4(:,71)=[]; BC6_5(:,71)=[]; BC6_6(:,71)=[]; BC6_7(:,71)=[];
% BC7_1(:,71)=[]; BC7_2(:,71)=[]; BC7_3(:,71)=[]; BC7_4(:,71)=[]; BC7_5(:,71)=[]; BC7_6(:,71)=[]; BC7_7(:,71)=[];
% BC8_1(:,71)=[]; BC8_2(:,71)=[]; BC8_3(:,71)=[]; BC8_4(:,71)=[]; BC8_5(:,71)=[]; BC8_6(:,71)=[]; BC8_7(:,71)=[];
% 
% %除去训练集的测试集
% BC_test1_1=a1_1;BC_test1_1(idx,:)=[];BC_test1_2=a1_2;BC_test1_2(idx,:)=[];BC_test1_3=a1_3;BC_test1_3(idx,:)=[];BC_test1_4=a1_4;BC_test1_4(idx,:)=[];
% BC_test1_5=a1_5;BC_test1_5(idx,:)=[];BC_test1_6=a1_6;BC_test1_6(idx,:)=[];BC_test1_7=a1_7;BC_test1_7(idx,:)=[];
% 
% BC_test2_1=a2_1;BC_test2_1(idx,:)=[];BC_test2_2=a2_2;BC_test2_2(idx,:)=[];BC_test2_3=a2_3;BC_test2_3(idx,:)=[];BC_test2_4=a2_4;BC_test2_4(idx,:)=[];
% BC_test2_5=a2_5;BC_test2_5(idx,:)=[];BC_test2_6=a2_6;BC_test2_6(idx,:)=[];BC_test2_7=a2_7;BC_test2_7(idx,:)=[];
% 
% BC_test3_1=a3_1;BC_test3_1(idx,:)=[];BC_test3_2=a3_2;BC_test3_2(idx,:)=[];BC_test3_3=a3_3;BC_test3_3(idx,:)=[];BC_test3_4=a3_4;BC_test3_4(idx,:)=[];
% BC_test3_5=a3_5;BC_test3_5(idx,:)=[];BC_test3_6=a3_6;BC_test3_6(idx,:)=[];BC_test3_7=a3_7;BC_test3_7(idx,:)=[];
% 
% BC_test4_1=a4_1;BC_test4_1(idx,:)=[];BC_test4_2=a4_2;BC_test4_2(idx,:)=[];BC_test4_3=a4_3;BC_test4_3(idx,:)=[];BC_test4_4=a4_4;BC_test4_4(idx,:)=[];
% BC_test4_5=a4_5;BC_test4_5(idx,:)=[];BC_test4_6=a4_6;BC_test4_6(idx,:)=[];BC_test4_7=a4_7;BC_test4_7(idx,:)=[];
% 
% BC_test5_1=a5_1;BC_test5_1(idx,:)=[];BC_test5_2=a5_2;BC_test5_2(idx,:)=[];BC_test5_3=a5_3;BC_test5_3(idx,:)=[];BC_test5_4=a5_4;BC_test5_4(idx,:)=[];
% BC_test5_5=a5_5;BC_test5_5(idx,:)=[];BC_test5_6=a5_6;BC_test5_6(idx,:)=[];BC_test5_7=a5_7;BC_test5_7(idx,:)=[];
% 
% BC_test6_1=a6_1;BC_test6_1(idx,:)=[];BC_test6_2=a6_2;BC_test6_2(idx,:)=[];BC_test6_3=a6_3;BC_test6_3(idx,:)=[];BC_test6_4=a6_4;BC_test6_4(idx,:)=[];
% BC_test6_5=a6_5;BC_test6_5(idx,:)=[];BC_test6_6=a6_6;BC_test6_6(idx,:)=[];BC_test6_7=a6_7;BC_test6_7(idx,:)=[];
% 
% BC_test7_1=a7_1;BC_test7_1(idx,:)=[];BC_test7_2=a7_2;BC_test7_2(idx,:)=[];BC_test7_3=a7_3;BC_test7_3(idx,:)=[];BC_test7_4=a7_4;BC_test7_4(idx,:)=[];
% BC_test7_5=a7_5;BC_test7_5(idx,:)=[];BC_test7_6=a7_6;BC_test7_6(idx,:)=[];BC_test7_7=a7_7;BC_test7_7(idx,:)=[];
% 
% BC_test8_1=a8_1;BC_test8_1(idx,:)=[];BC_test8_2=a8_2;BC_test8_2(idx,:)=[];BC_test8_3=a8_3;BC_test8_3(idx,:)=[];BC_test8_4=a8_4;BC_test8_4(idx,:)=[];
% BC_test8_5=a8_5;BC_test8_5(idx,:)=[];BC_test8_6=a8_6;BC_test8_6(idx,:)=[];BC_test8_7=a8_7;BC_test8_7(idx,:)=[];
% 
% %7个基分类器测试集标签
% y_test_BC1_1=BC_test1_1(:,71);y_test_BC1_2=BC_test1_2(:,71);y_test_BC1_3=BC_test1_3(:,71);y_test_BC1_4=BC_test1_4(:,71);y_test_BC1_5=BC_test1_5(:,71);y_test_BC1_6=BC_test1_6(:,71);y_test_BC1_7=BC_test1_7(:,71);
% y_test_BC2_1=BC_test2_1(:,71);y_test_BC2_2=BC_test1_2(:,71);y_test_BC2_3=BC_test2_3(:,71);y_test_BC2_4=BC_test2_4(:,71);y_test_BC2_5=BC_test2_5(:,71);y_test_BC2_6=BC_test2_6(:,71);y_test_BC2_7=BC_test2_7(:,71);
% y_test_BC3_1=BC_test3_1(:,71);y_test_BC3_2=BC_test3_2(:,71);y_test_BC3_3=BC_test3_3(:,71);y_test_BC3_4=BC_test3_4(:,71);y_test_BC3_5=BC_test3_5(:,71);y_test_BC3_6=BC_test3_6(:,71);y_test_BC3_7=BC_test3_7(:,71);
% y_test_BC4_1=BC_test4_1(:,71);y_test_BC4_2=BC_test4_2(:,71);y_test_BC4_3=BC_test4_3(:,71);y_test_BC4_4=BC_test4_4(:,71);y_test_BC4_5=BC_test4_5(:,71);y_test_BC4_6=BC_test4_6(:,71);y_test_BC4_7=BC_test4_7(:,71);
% y_test_BC5_1=BC_test5_1(:,71);y_test_BC5_2=BC_test5_2(:,71);y_test_BC5_3=BC_test5_3(:,71);y_test_BC5_4=BC_test5_4(:,71);y_test_BC5_5=BC_test5_5(:,71);y_test_BC5_6=BC_test5_6(:,71);y_test_BC5_7=BC_test5_7(:,71);
% y_test_BC6_1=BC_test6_1(:,71);y_test_BC6_2=BC_test6_2(:,71);y_test_BC6_3=BC_test6_3(:,71);y_test_BC6_4=BC_test6_4(:,71);y_test_BC6_5=BC_test6_5(:,71);y_test_BC6_6=BC_test6_6(:,71);y_test_BC6_7=BC_test6_7(:,71);
% y_test_BC7_1=BC_test7_1(:,71);y_test_BC7_2=BC_test7_2(:,71);y_test_BC7_3=BC_test7_3(:,71);y_test_BC7_4=BC_test7_4(:,71);y_test_BC7_5=BC_test7_5(:,71);y_test_BC7_6=BC_test7_6(:,71);y_test_BC7_7=BC_test7_7(:,71);
% y_test_BC8_1=BC_test8_1(:,71);y_test_BC8_2=BC_test8_2(:,71);y_test_BC8_3=BC_test8_3(:,71);y_test_BC8_4=BC_test8_4(:,71);y_test_BC8_5=BC_test8_5(:,71);y_test_BC8_6=BC_test8_6(:,71);y_test_BC8_7=BC_test8_7(:,71);
% %7个基分类器测试集标签，按列排
% y_test_BC1=[y_test_BC1_1;y_test_BC1_2;y_test_BC1_3;y_test_BC1_4;y_test_BC1_5;y_test_BC1_6;y_test_BC1_7];
% y_test_BC2=[y_test_BC2_1;y_test_BC2_2;y_test_BC2_3;y_test_BC2_4;y_test_BC2_5;y_test_BC2_6;y_test_BC2_7];
% y_test_BC3=[y_test_BC3_1;y_test_BC3_2;y_test_BC3_3;y_test_BC3_4;y_test_BC3_5;y_test_BC3_6;y_test_BC3_7];
% y_test_BC4=[y_test_BC4_1;y_test_BC4_2;y_test_BC4_3;y_test_BC4_4;y_test_BC4_5;y_test_BC4_6;y_test_BC4_7];
% y_test_BC5=[y_test_BC5_1;y_test_BC5_2;y_test_BC5_3;y_test_BC5_4;y_test_BC5_5;y_test_BC5_6;y_test_BC5_7];
% y_test_BC6=[y_test_BC6_1;y_test_BC6_2;y_test_BC6_3;y_test_BC6_4;y_test_BC6_5;y_test_BC6_6;y_test_BC6_7];
% y_test_BC7=[y_test_BC7_1;y_test_BC7_2;y_test_BC7_3;y_test_BC7_4;y_test_BC7_5;y_test_BC7_6;y_test_BC7_7];
% y_test_BC8=[y_test_BC8_1;y_test_BC8_2;y_test_BC8_3;y_test_BC8_4;y_test_BC8_5;y_test_BC8_6;y_test_BC8_7];
% 
% %分别取7位被试的测试集
% BC_test1_1(:,71)=[]; BC_test1_2(:,71)=[]; BC_test1_3(:,71)=[]; BC_test1_4(:,71)=[]; BC_test1_5(:,71)=[]; BC_test1_6(:,71)=[]; BC_test1_7(:,71)=[];
% BC_test2_1(:,71)=[]; BC_test2_2(:,71)=[]; BC_test2_3(:,71)=[]; BC_test2_4(:,71)=[]; BC_test2_5(:,71)=[]; BC_test2_6(:,71)=[]; BC_test2_7(:,71)=[];
% BC_test3_1(:,71)=[]; BC_test3_2(:,71)=[]; BC_test3_3(:,71)=[]; BC_test3_4(:,71)=[]; BC_test3_5(:,71)=[]; BC_test3_6(:,71)=[]; BC_test3_7(:,71)=[];
% BC_test4_1(:,71)=[]; BC_test4_2(:,71)=[]; BC_test4_3(:,71)=[]; BC_test4_4(:,71)=[]; BC_test4_5(:,71)=[]; BC_test4_6(:,71)=[]; BC_test4_7(:,71)=[];
% BC_test5_1(:,71)=[]; BC_test5_2(:,71)=[]; BC_test5_3(:,71)=[]; BC_test5_4(:,71)=[]; BC_test5_5(:,71)=[]; BC_test5_6(:,71)=[]; BC_test5_7(:,71)=[];
% BC_test6_1(:,71)=[]; BC_test6_2(:,71)=[]; BC_test6_3(:,71)=[]; BC_test6_4(:,71)=[]; BC_test6_5(:,71)=[]; BC_test6_6(:,71)=[]; BC_test6_7(:,71)=[];
% BC_test7_1(:,71)=[]; BC_test7_2(:,71)=[]; BC_test7_3(:,71)=[]; BC_test7_4(:,71)=[]; BC_test7_5(:,71)=[]; BC_test7_6(:,71)=[]; BC_test7_7(:,71)=[];
% BC_test8_1(:,71)=[]; BC_test8_2(:,71)=[]; BC_test8_3(:,71)=[]; BC_test8_4(:,71)=[]; BC_test8_5(:,71)=[]; BC_test8_6(:,71)=[]; BC_test8_7(:,71)=[];
% 
% %基分类器的测试集，7个基分类器对同一数据测试，得出并序7组数据，然后通过Q-statistic，选取最佳的几排预测结果
% BC_test1=[BC_test1_1;BC_test1_2;BC_test1_3;BC_test1_4;BC_test1_5;BC_test1_6;BC_test1_7];
% BC_test2=[BC_test2_1;BC_test2_2;BC_test2_3;BC_test2_4;BC_test2_5;BC_test2_6;BC_test2_7];
% BC_test3=[BC_test3_1;BC_test3_2;BC_test3_3;BC_test3_4;BC_test3_5;BC_test3_6;BC_test3_7];
% BC_test4=[BC_test4_1;BC_test4_2;BC_test4_3;BC_test4_4;BC_test4_5;BC_test4_6;BC_test4_7];
% BC_test5=[BC_test5_1;BC_test5_2;BC_test5_3;BC_test5_4;BC_test5_5;BC_test5_6;BC_test5_7];
% BC_test6=[BC_test6_1;BC_test6_2;BC_test6_3;BC_test6_4;BC_test6_5;BC_test6_6;BC_test6_7];
% BC_test7=[BC_test7_1;BC_test7_2;BC_test7_3;BC_test7_4;BC_test7_5;BC_test7_6;BC_test7_7];
% BC_test8=[BC_test8_1;BC_test8_2;BC_test8_3;BC_test8_4;BC_test8_5;BC_test8_6;BC_test8_7];
% 
% %次级分类器测试集数据输入准备，给8个ELM测试
% b1=cell2mat(xte_all_1(1,:));
% b2=cell2mat(xte_all_1(2,:));
% b3=cell2mat(xte_all_1(3,:));
% b4=cell2mat(xte_all_1(4,:));
% b5=cell2mat(xte_all_1(5,:));
% b6=cell2mat(xte_all_1(6,:));
% b7=cell2mat(xte_all_1(7,:));
% b8=cell2mat(xte_all_1(8,:));
% 
% %次级分类器测试集，做跨被试，分别做8次测试，得出8组结果，然后平均，才为任务一阶段1的平均结果
% y_test_DC=y;  %每位被试的标签都一样
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session1\BC_test BC_test1 BC_test2 BC_test3 BC_test4 BC_test5 BC_test6 BC_test7 BC_test8...
%      y_test_BC1 y_test_BC2 y_test_BC3 y_test_BC4 y_test_BC5 y_test_BC6 y_test_BC7 y_test_BC8
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session1\BC_train BC1_1 BC1_2 BC1_3 BC1_4 BC1_5 BC1_6 BC1_7...
%      BC2_1 BC2_2 BC2_3 BC2_4 BC2_5 BC2_6 BC2_7 BC3_1 BC3_2 BC3_3 BC3_4 BC3_5 BC3_6 BC3_7 BC4_1 BC4_2 BC4_3 BC4_4 BC4_5 BC4_6 BC4_7...
%      BC5_1 BC5_2 BC5_3 BC5_4 BC5_5 BC5_6 BC5_7 BC6_1 BC6_2 BC6_3 BC6_4 BC6_5 BC6_6 BC6_7 BC7_1 BC7_2 BC7_3 BC7_4 BC7_5 BC7_6 BC7_7...
%      BC8_1 BC8_2 BC8_3 BC8_4 BC8_5 BC8_6 BC8_7...
%      y_train_BC1_1 y_train_BC1_2 y_train_BC1_3 y_train_BC1_4 y_train_BC1_5 y_train_BC1_6 y_train_BC1_7...
%      y_train_BC2_1 y_train_BC2_2 y_train_BC2_3 y_train_BC2_4 y_train_BC2_5 y_train_BC2_6 y_train_BC2_7...
%      y_train_BC3_1 y_train_BC3_2 y_train_BC3_3 y_train_BC3_4 y_train_BC3_5 y_train_BC3_6 y_train_BC3_7...
%      y_train_BC4_1 y_train_BC4_2 y_train_BC4_3 y_train_BC4_4 y_train_BC4_5 y_train_BC4_6 y_train_BC4_7...
%      y_train_BC5_1 y_train_BC5_2 y_train_BC5_3 y_train_BC5_4 y_train_BC5_5 y_train_BC5_6 y_train_BC5_7...
%      y_train_BC6_1 y_train_BC6_2 y_train_BC6_3 y_train_BC6_4 y_train_BC6_5 y_train_BC6_6 y_train_BC6_7...
%      y_train_BC7_1 y_train_BC7_2 y_train_BC7_3 y_train_BC7_4 y_train_BC7_5 y_train_BC7_6 y_train_BC7_7...
%      y_train_BC8_1 y_train_BC8_2 y_train_BC8_3 y_train_BC8_4 y_train_BC8_5 y_train_BC8_6 y_train_BC8_7
%  save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session1\DC_tset b1 b2 b3 b4 b5 b6 b7 b8 y_test_DC
 
% %case1_session2
% %基分类器数据输入准备
% a1=cell2mat(xtr_all_2(1,:));
% a2=cell2mat(xtr_all_2(2,:));
% a3=cell2mat(xtr_all_2(3,:));
% a4=cell2mat(xtr_all_2(4,:));
% a5=cell2mat(xtr_all_2(5,:));
% a6=cell2mat(xtr_all_2(6,:));
% a7=cell2mat(xtr_all_2(7,:));
% a8=cell2mat(xtr_all_2(8,:));
% 
% %输入数据附标签
% a1(:,71)=repmat(y,7,1);
% a2(:,71)=repmat(y,7,1);
% a3(:,71)=repmat(y,7,1);
% a4(:,71)=repmat(y,7,1);
% a5(:,71)=repmat(y,7,1);
% a6(:,71)=repmat(y,7,1);
% a7(:,71)=repmat(y,7,1);
% a8(:,71)=repmat(y,7,1);
% 
% %剩余7位被试分开
% a1_1=a1(1:2700,:); a1_2=a1(2701:5400,:); a1_3=a1(5401:8100,:); a1_4=a1(8101:10800,:); a1_5=a1(10801:13500,:); a1_6=a1(13501:16200,:); a1_7=a1(16201:18900,:);
% a2_1=a2(1:2700,:); a2_2=a2(2701:5400,:); a2_3=a2(5401:8100,:); a2_4=a2(8101:10800,:); a2_5=a2(10801:13500,:); a2_6=a2(13501:16200,:); a2_7=a2(16201:18900,:);
% a3_1=a3(1:2700,:); a3_2=a3(2701:5400,:); a3_3=a3(5401:8100,:); a3_4=a3(8101:10800,:); a3_5=a3(10801:13500,:); a3_6=a3(13501:16200,:); a3_7=a3(16201:18900,:);
% a4_1=a4(1:2700,:); a4_2=a4(2701:5400,:); a4_3=a4(5401:8100,:); a4_4=a4(8101:10800,:); a4_5=a4(10801:13500,:); a4_6=a4(13501:16200,:); a4_7=a4(16201:18900,:);
% a5_1=a5(1:2700,:); a5_2=a5(2701:5400,:); a5_3=a5(5401:8100,:); a5_4=a5(8101:10800,:); a5_5=a5(10801:13500,:); a5_6=a5(13501:16200,:); a5_7=a5(16201:18900,:);
% a6_1=a6(1:2700,:); a6_2=a6(2701:5400,:); a6_3=a6(5401:8100,:); a6_4=a6(8101:10800,:); a6_5=a6(10801:13500,:); a6_6=a6(13501:16200,:); a6_7=a6(16201:18900,:);
% a7_1=a7(1:2700,:); a7_2=a7(2701:5400,:); a7_3=a7(5401:8100,:); a7_4=a7(8101:10800,:); a7_5=a7(10801:13500,:); a7_6=a7(13501:16200,:); a7_7=a7(16201:18900,:);
% a8_1=a8(1:2700,:); a8_2=a8(2701:5400,:); a8_3=a8(5401:8100,:); a8_4=a8(8101:10800,:); a8_5=a8(10801:13500,:); a8_6=a8(13501:16200,:); a8_7=a8(16201:18900,:);
% 
% %分别取7位被试的训练集
% n=2600; idx=randperm(2700); idx=idx(1:n);
% BC1_1=a1_1(idx,:); BC1_2=a1_2(idx,:); BC1_3=a1_3(idx,:); BC1_4=a1_4(idx,:); BC1_5=a1_5(idx,:); BC1_6=a1_6(idx,:); BC1_7=a1_7(idx,:);
% BC2_1=a2_1(idx,:); BC2_2=a2_2(idx,:); BC2_3=a2_3(idx,:); BC2_4=a2_4(idx,:); BC2_5=a2_5(idx,:); BC2_6=a2_6(idx,:); BC2_7=a2_7(idx,:);
% BC3_1=a3_1(idx,:); BC3_2=a3_2(idx,:); BC3_3=a3_3(idx,:); BC3_4=a3_4(idx,:); BC3_5=a3_5(idx,:); BC3_6=a3_6(idx,:); BC3_7=a3_7(idx,:);
% BC4_1=a4_1(idx,:); BC4_2=a4_2(idx,:); BC4_3=a4_3(idx,:); BC4_4=a4_4(idx,:); BC4_5=a4_5(idx,:); BC4_6=a4_6(idx,:); BC4_7=a4_7(idx,:);
% BC5_1=a5_1(idx,:); BC5_2=a5_2(idx,:); BC5_3=a5_3(idx,:); BC5_4=a5_4(idx,:); BC5_5=a5_5(idx,:); BC5_6=a5_6(idx,:); BC5_7=a5_7(idx,:);
% BC6_1=a6_1(idx,:); BC6_2=a6_2(idx,:); BC6_3=a6_3(idx,:); BC6_4=a6_4(idx,:); BC6_5=a6_5(idx,:); BC6_6=a6_6(idx,:); BC6_7=a6_7(idx,:);
% BC7_1=a7_1(idx,:); BC7_2=a7_2(idx,:); BC7_3=a7_3(idx,:); BC7_4=a7_4(idx,:); BC7_5=a7_5(idx,:); BC7_6=a7_6(idx,:); BC7_7=a7_7(idx,:);
% BC8_1=a8_1(idx,:); BC8_2=a8_2(idx,:); BC8_3=a8_3(idx,:); BC8_4=a8_4(idx,:); BC8_5=a8_5(idx,:); BC8_6=a8_6(idx,:); BC8_7=a8_7(idx,:);
% %7个基分类器训练集标签
% y_train_BC1_1=BC1_1(:,71);y_train_BC1_2=BC1_2(:,71);y_train_BC1_3=BC1_3(:,71);y_train_BC1_4=BC1_4(:,71);y_train_BC1_5=BC1_5(:,71);y_train_BC1_6=BC1_6(:,71);y_train_BC1_7=BC1_7(:,71);
% y_train_BC2_1=BC2_1(:,71);y_train_BC2_2=BC2_2(:,71);y_train_BC2_3=BC2_3(:,71);y_train_BC2_4=BC2_4(:,71);y_train_BC2_5=BC2_5(:,71);y_train_BC2_6=BC2_6(:,71);y_train_BC2_7=BC2_7(:,71);
% y_train_BC3_1=BC3_1(:,71);y_train_BC3_2=BC3_2(:,71);y_train_BC3_3=BC3_3(:,71);y_train_BC3_4=BC3_4(:,71);y_train_BC3_5=BC3_5(:,71);y_train_BC3_6=BC3_6(:,71);y_train_BC3_7=BC3_7(:,71);
% y_train_BC4_1=BC4_1(:,71);y_train_BC4_2=BC4_2(:,71);y_train_BC4_3=BC4_3(:,71);y_train_BC4_4=BC4_4(:,71);y_train_BC4_5=BC4_5(:,71);y_train_BC4_6=BC4_6(:,71);y_train_BC4_7=BC4_7(:,71);
% y_train_BC5_1=BC5_1(:,71);y_train_BC5_2=BC5_1(:,71);y_train_BC5_3=BC5_3(:,71);y_train_BC5_4=BC5_4(:,71);y_train_BC5_5=BC5_5(:,71);y_train_BC5_6=BC5_6(:,71);y_train_BC5_7=BC5_7(:,71);
% y_train_BC6_1=BC6_1(:,71);y_train_BC6_2=BC6_2(:,71);y_train_BC6_3=BC6_3(:,71);y_train_BC6_4=BC6_4(:,71);y_train_BC6_5=BC6_5(:,71);y_train_BC6_6=BC6_6(:,71);y_train_BC6_7=BC6_7(:,71);
% y_train_BC7_1=BC7_1(:,71);y_train_BC7_2=BC7_2(:,71);y_train_BC7_3=BC7_3(:,71);y_train_BC7_4=BC7_4(:,71);y_train_BC7_5=BC7_5(:,71);y_train_BC7_6=BC7_6(:,71);y_train_BC7_7=BC7_7(:,71);
% y_train_BC8_1=BC8_1(:,71);y_train_BC8_2=BC8_2(:,71);y_train_BC8_3=BC8_3(:,71);y_train_BC8_4=BC8_4(:,71);y_train_BC8_5=BC8_5(:,71);y_train_BC8_6=BC8_6(:,71);y_train_BC8_7=BC8_7(:,71);
% 
% %%%%
% %7个基分类器训练集
% BC1_1(:,71)=[]; BC1_2(:,71)=[]; BC1_3(:,71)=[]; BC1_4(:,71)=[]; BC1_5(:,71)=[]; BC1_6(:,71)=[]; BC1_7(:,71)=[];
% BC2_1(:,71)=[]; BC2_2(:,71)=[]; BC2_3(:,71)=[]; BC2_4(:,71)=[]; BC2_5(:,71)=[]; BC2_6(:,71)=[]; BC2_7(:,71)=[];
% BC3_1(:,71)=[]; BC3_2(:,71)=[]; BC3_3(:,71)=[]; BC3_4(:,71)=[]; BC3_5(:,71)=[]; BC3_6(:,71)=[]; BC3_7(:,71)=[];
% BC4_1(:,71)=[]; BC4_2(:,71)=[]; BC4_3(:,71)=[]; BC4_4(:,71)=[]; BC4_5(:,71)=[]; BC4_6(:,71)=[]; BC4_7(:,71)=[];
% BC5_1(:,71)=[]; BC5_2(:,71)=[]; BC5_3(:,71)=[]; BC5_4(:,71)=[]; BC5_5(:,71)=[]; BC5_6(:,71)=[]; BC5_7(:,71)=[];
% BC6_1(:,71)=[]; BC6_2(:,71)=[]; BC6_3(:,71)=[]; BC6_4(:,71)=[]; BC6_5(:,71)=[]; BC6_6(:,71)=[]; BC6_7(:,71)=[];
% BC7_1(:,71)=[]; BC7_2(:,71)=[]; BC7_3(:,71)=[]; BC7_4(:,71)=[]; BC7_5(:,71)=[]; BC7_6(:,71)=[]; BC7_7(:,71)=[];
% BC8_1(:,71)=[]; BC8_2(:,71)=[]; BC8_3(:,71)=[]; BC8_4(:,71)=[]; BC8_5(:,71)=[]; BC8_6(:,71)=[]; BC8_7(:,71)=[];
% 
% %除去训练集的测试集
% BC_test1_1=a1_1;BC_test1_1(idx,:)=[];BC_test1_2=a1_2;BC_test1_2(idx,:)=[];BC_test1_3=a1_3;BC_test1_3(idx,:)=[];BC_test1_4=a1_4;BC_test1_4(idx,:)=[];
% BC_test1_5=a1_5;BC_test1_5(idx,:)=[];BC_test1_6=a1_6;BC_test1_6(idx,:)=[];BC_test1_7=a1_7;BC_test1_7(idx,:)=[];
% 
% BC_test2_1=a2_1;BC_test2_1(idx,:)=[];BC_test2_2=a2_2;BC_test2_2(idx,:)=[];BC_test2_3=a2_3;BC_test2_3(idx,:)=[];BC_test2_4=a2_4;BC_test2_4(idx,:)=[];
% BC_test2_5=a2_5;BC_test2_5(idx,:)=[];BC_test2_6=a2_6;BC_test2_6(idx,:)=[];BC_test2_7=a2_7;BC_test2_7(idx,:)=[];
% 
% BC_test3_1=a3_1;BC_test3_1(idx,:)=[];BC_test3_2=a3_2;BC_test3_2(idx,:)=[];BC_test3_3=a3_3;BC_test3_3(idx,:)=[];BC_test3_4=a3_4;BC_test3_4(idx,:)=[];
% BC_test3_5=a3_5;BC_test3_5(idx,:)=[];BC_test3_6=a3_6;BC_test3_6(idx,:)=[];BC_test3_7=a3_7;BC_test3_7(idx,:)=[];
% 
% BC_test4_1=a4_1;BC_test4_1(idx,:)=[];BC_test4_2=a4_2;BC_test4_2(idx,:)=[];BC_test4_3=a4_3;BC_test4_3(idx,:)=[];BC_test4_4=a4_4;BC_test4_4(idx,:)=[];
% BC_test4_5=a4_5;BC_test4_5(idx,:)=[];BC_test4_6=a4_6;BC_test4_6(idx,:)=[];BC_test4_7=a4_7;BC_test4_7(idx,:)=[];
% 
% BC_test5_1=a5_1;BC_test5_1(idx,:)=[];BC_test5_2=a5_2;BC_test5_2(idx,:)=[];BC_test5_3=a5_3;BC_test5_3(idx,:)=[];BC_test5_4=a5_4;BC_test5_4(idx,:)=[];
% BC_test5_5=a5_5;BC_test5_5(idx,:)=[];BC_test5_6=a5_6;BC_test5_6(idx,:)=[];BC_test5_7=a5_7;BC_test5_7(idx,:)=[];
% 
% BC_test6_1=a6_1;BC_test6_1(idx,:)=[];BC_test6_2=a6_2;BC_test6_2(idx,:)=[];BC_test6_3=a6_3;BC_test6_3(idx,:)=[];BC_test6_4=a6_4;BC_test6_4(idx,:)=[];
% BC_test6_5=a6_5;BC_test6_5(idx,:)=[];BC_test6_6=a6_6;BC_test6_6(idx,:)=[];BC_test6_7=a6_7;BC_test6_7(idx,:)=[];
% 
% BC_test7_1=a7_1;BC_test7_1(idx,:)=[];BC_test7_2=a7_2;BC_test7_2(idx,:)=[];BC_test7_3=a7_3;BC_test7_3(idx,:)=[];BC_test7_4=a7_4;BC_test7_4(idx,:)=[];
% BC_test7_5=a7_5;BC_test7_5(idx,:)=[];BC_test7_6=a7_6;BC_test7_6(idx,:)=[];BC_test7_7=a7_7;BC_test7_7(idx,:)=[];
% 
% BC_test8_1=a8_1;BC_test8_1(idx,:)=[];BC_test8_2=a8_2;BC_test8_2(idx,:)=[];BC_test8_3=a8_3;BC_test8_3(idx,:)=[];BC_test8_4=a8_4;BC_test8_4(idx,:)=[];
% BC_test8_5=a8_5;BC_test8_5(idx,:)=[];BC_test8_6=a8_6;BC_test8_6(idx,:)=[];BC_test8_7=a8_7;BC_test8_7(idx,:)=[];
% 
% %7个基分类器测试集标签
% y_test_BC1_1=BC_test1_1(:,71);y_test_BC1_2=BC_test1_2(:,71);y_test_BC1_3=BC_test1_3(:,71);y_test_BC1_4=BC_test1_4(:,71);y_test_BC1_5=BC_test1_5(:,71);y_test_BC1_6=BC_test1_6(:,71);y_test_BC1_7=BC_test1_7(:,71);
% y_test_BC2_1=BC_test2_1(:,71);y_test_BC2_2=BC_test1_2(:,71);y_test_BC2_3=BC_test2_3(:,71);y_test_BC2_4=BC_test2_4(:,71);y_test_BC2_5=BC_test2_5(:,71);y_test_BC2_6=BC_test2_6(:,71);y_test_BC2_7=BC_test2_7(:,71);
% y_test_BC3_1=BC_test3_1(:,71);y_test_BC3_2=BC_test3_2(:,71);y_test_BC3_3=BC_test3_3(:,71);y_test_BC3_4=BC_test3_4(:,71);y_test_BC3_5=BC_test3_5(:,71);y_test_BC3_6=BC_test3_6(:,71);y_test_BC3_7=BC_test3_7(:,71);
% y_test_BC4_1=BC_test4_1(:,71);y_test_BC4_2=BC_test4_2(:,71);y_test_BC4_3=BC_test4_3(:,71);y_test_BC4_4=BC_test4_4(:,71);y_test_BC4_5=BC_test4_5(:,71);y_test_BC4_6=BC_test4_6(:,71);y_test_BC4_7=BC_test4_7(:,71);
% y_test_BC5_1=BC_test5_1(:,71);y_test_BC5_2=BC_test5_2(:,71);y_test_BC5_3=BC_test5_3(:,71);y_test_BC5_4=BC_test5_4(:,71);y_test_BC5_5=BC_test5_5(:,71);y_test_BC5_6=BC_test5_6(:,71);y_test_BC5_7=BC_test5_7(:,71);
% y_test_BC6_1=BC_test6_1(:,71);y_test_BC6_2=BC_test6_2(:,71);y_test_BC6_3=BC_test6_3(:,71);y_test_BC6_4=BC_test6_4(:,71);y_test_BC6_5=BC_test6_5(:,71);y_test_BC6_6=BC_test6_6(:,71);y_test_BC6_7=BC_test6_7(:,71);
% y_test_BC7_1=BC_test7_1(:,71);y_test_BC7_2=BC_test7_2(:,71);y_test_BC7_3=BC_test7_3(:,71);y_test_BC7_4=BC_test7_4(:,71);y_test_BC7_5=BC_test7_5(:,71);y_test_BC7_6=BC_test7_6(:,71);y_test_BC7_7=BC_test7_7(:,71);
% y_test_BC8_1=BC_test8_1(:,71);y_test_BC8_2=BC_test8_2(:,71);y_test_BC8_3=BC_test8_3(:,71);y_test_BC8_4=BC_test8_4(:,71);y_test_BC8_5=BC_test8_5(:,71);y_test_BC8_6=BC_test8_6(:,71);y_test_BC8_7=BC_test8_7(:,71);
% %7个基分类器测试集标签，按列排
% y_test_BC1=[y_test_BC1_1;y_test_BC1_2;y_test_BC1_3;y_test_BC1_4;y_test_BC1_5;y_test_BC1_6;y_test_BC1_7];
% y_test_BC2=[y_test_BC2_1;y_test_BC2_2;y_test_BC2_3;y_test_BC2_4;y_test_BC2_5;y_test_BC2_6;y_test_BC2_7];
% y_test_BC3=[y_test_BC3_1;y_test_BC3_2;y_test_BC3_3;y_test_BC3_4;y_test_BC3_5;y_test_BC3_6;y_test_BC3_7];
% y_test_BC4=[y_test_BC4_1;y_test_BC4_2;y_test_BC4_3;y_test_BC4_4;y_test_BC4_5;y_test_BC4_6;y_test_BC4_7];
% y_test_BC5=[y_test_BC5_1;y_test_BC5_2;y_test_BC5_3;y_test_BC5_4;y_test_BC5_5;y_test_BC5_6;y_test_BC5_7];
% y_test_BC6=[y_test_BC6_1;y_test_BC6_2;y_test_BC6_3;y_test_BC6_4;y_test_BC6_5;y_test_BC6_6;y_test_BC6_7];
% y_test_BC7=[y_test_BC7_1;y_test_BC7_2;y_test_BC7_3;y_test_BC7_4;y_test_BC7_5;y_test_BC7_6;y_test_BC7_7];
% y_test_BC8=[y_test_BC8_1;y_test_BC8_2;y_test_BC8_3;y_test_BC8_4;y_test_BC8_5;y_test_BC8_6;y_test_BC8_7];
% 
% %分别取7位被试的测试集
% BC_test1_1(:,71)=[]; BC_test1_2(:,71)=[]; BC_test1_3(:,71)=[]; BC_test1_4(:,71)=[]; BC_test1_5(:,71)=[]; BC_test1_6(:,71)=[]; BC_test1_7(:,71)=[];
% BC_test2_1(:,71)=[]; BC_test2_2(:,71)=[]; BC_test2_3(:,71)=[]; BC_test2_4(:,71)=[]; BC_test2_5(:,71)=[]; BC_test2_6(:,71)=[]; BC_test2_7(:,71)=[];
% BC_test3_1(:,71)=[]; BC_test3_2(:,71)=[]; BC_test3_3(:,71)=[]; BC_test3_4(:,71)=[]; BC_test3_5(:,71)=[]; BC_test3_6(:,71)=[]; BC_test3_7(:,71)=[];
% BC_test4_1(:,71)=[]; BC_test4_2(:,71)=[]; BC_test4_3(:,71)=[]; BC_test4_4(:,71)=[]; BC_test4_5(:,71)=[]; BC_test4_6(:,71)=[]; BC_test4_7(:,71)=[];
% BC_test5_1(:,71)=[]; BC_test5_2(:,71)=[]; BC_test5_3(:,71)=[]; BC_test5_4(:,71)=[]; BC_test5_5(:,71)=[]; BC_test5_6(:,71)=[]; BC_test5_7(:,71)=[];
% BC_test6_1(:,71)=[]; BC_test6_2(:,71)=[]; BC_test6_3(:,71)=[]; BC_test6_4(:,71)=[]; BC_test6_5(:,71)=[]; BC_test6_6(:,71)=[]; BC_test6_7(:,71)=[];
% BC_test7_1(:,71)=[]; BC_test7_2(:,71)=[]; BC_test7_3(:,71)=[]; BC_test7_4(:,71)=[]; BC_test7_5(:,71)=[]; BC_test7_6(:,71)=[]; BC_test7_7(:,71)=[];
% BC_test8_1(:,71)=[]; BC_test8_2(:,71)=[]; BC_test8_3(:,71)=[]; BC_test8_4(:,71)=[]; BC_test8_5(:,71)=[]; BC_test8_6(:,71)=[]; BC_test8_7(:,71)=[];
% 
% %基分类器的测试集，7个基分类器对同一数据测试，得出并序7组数据，然后通过Q-statistic，选取最佳的几排预测结果
% BC_test1=[BC_test1_1;BC_test1_2;BC_test1_3;BC_test1_4;BC_test1_5;BC_test1_6;BC_test1_7];
% BC_test2=[BC_test2_1;BC_test2_2;BC_test2_3;BC_test2_4;BC_test2_5;BC_test2_6;BC_test2_7];
% BC_test3=[BC_test3_1;BC_test3_2;BC_test3_3;BC_test3_4;BC_test3_5;BC_test3_6;BC_test3_7];
% BC_test4=[BC_test4_1;BC_test4_2;BC_test4_3;BC_test4_4;BC_test4_5;BC_test4_6;BC_test4_7];
% BC_test5=[BC_test5_1;BC_test5_2;BC_test5_3;BC_test5_4;BC_test5_5;BC_test5_6;BC_test5_7];
% BC_test6=[BC_test6_1;BC_test6_2;BC_test6_3;BC_test6_4;BC_test6_5;BC_test6_6;BC_test6_7];
% BC_test7=[BC_test7_1;BC_test7_2;BC_test7_3;BC_test7_4;BC_test7_5;BC_test7_6;BC_test7_7];
% BC_test8=[BC_test8_1;BC_test8_2;BC_test8_3;BC_test8_4;BC_test8_5;BC_test8_6;BC_test8_7];
% 
% %次级分类器测试集数据输入准备，给8个ELM测试
% b1=cell2mat(xte_all_2(1,:));
% b2=cell2mat(xte_all_2(2,:));
% b3=cell2mat(xte_all_2(3,:));
% b4=cell2mat(xte_all_2(4,:));
% b5=cell2mat(xte_all_2(5,:));
% b6=cell2mat(xte_all_2(6,:));
% b7=cell2mat(xte_all_2(7,:));
% b8=cell2mat(xte_all_2(8,:));
% 
% %次级分类器测试集，做跨被试，分别做8次测试，得出8组结果，然后平均，才为任务一阶段1的平均结果
% y_test_DC=y;  %每位被试的标签都一样
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session2\BC_test_2 BC_test1 BC_test2 BC_test3 BC_test4 BC_test5 BC_test6 BC_test7 BC_test8...
%      y_test_BC1 y_test_BC2 y_test_BC3 y_test_BC4 y_test_BC5 y_test_BC6 y_test_BC7 y_test_BC8
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session2\BC_train_2 BC1_1 BC1_2 BC1_3 BC1_4 BC1_5 BC1_6 BC1_7...
%      BC2_1 BC2_2 BC2_3 BC2_4 BC2_5 BC2_6 BC2_7 BC3_1 BC3_2 BC3_3 BC3_4 BC3_5 BC3_6 BC3_7 BC4_1 BC4_2 BC4_3 BC4_4 BC4_5 BC4_6 BC4_7...
%      BC5_1 BC5_2 BC5_3 BC5_4 BC5_5 BC5_6 BC5_7 BC6_1 BC6_2 BC6_3 BC6_4 BC6_5 BC6_6 BC6_7 BC7_1 BC7_2 BC7_3 BC7_4 BC7_5 BC7_6 BC7_7...
%      BC8_1 BC8_2 BC8_3 BC8_4 BC8_5 BC8_6 BC8_7...
%      y_train_BC1_1 y_train_BC1_2 y_train_BC1_3 y_train_BC1_4 y_train_BC1_5 y_train_BC1_6 y_train_BC1_7...
%      y_train_BC2_1 y_train_BC2_2 y_train_BC2_3 y_train_BC2_4 y_train_BC2_5 y_train_BC2_6 y_train_BC2_7...
%      y_train_BC3_1 y_train_BC3_2 y_train_BC3_3 y_train_BC3_4 y_train_BC3_5 y_train_BC3_6 y_train_BC3_7...
%      y_train_BC4_1 y_train_BC4_2 y_train_BC4_3 y_train_BC4_4 y_train_BC4_5 y_train_BC4_6 y_train_BC4_7...
%      y_train_BC5_1 y_train_BC5_2 y_train_BC5_3 y_train_BC5_4 y_train_BC5_5 y_train_BC5_6 y_train_BC5_7...
%      y_train_BC6_1 y_train_BC6_2 y_train_BC6_3 y_train_BC6_4 y_train_BC6_5 y_train_BC6_6 y_train_BC6_7...
%      y_train_BC7_1 y_train_BC7_2 y_train_BC7_3 y_train_BC7_4 y_train_BC7_5 y_train_BC7_6 y_train_BC7_7...
%      y_train_BC8_1 y_train_BC8_2 y_train_BC8_3 y_train_BC8_4 y_train_BC8_5 y_train_BC8_6 y_train_BC8_7
%  save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case1_session2\DC_test_2 b1 b2 b3 b4 b5 b6 b7 b8 y_test_DC

%case2
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

rr=70;
%进行LPP
fea =x33;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 0.8;
[eigvector, eigvalue] = LPP(W, options, fea);
x33=fea*eigvector;

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

%进行LPP
fea =x44;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 0.8;
[eigvector, eigvalue] = LPP(W, options, fea);
x44=fea*eigvector;

%任务二两个阶段的数据集
x3=x33;
x4=x44;

% y_1=[y;y;y;y;y;y];

K=6;
% low=size(find(y_1==1));
% medium=size(find(y_1==2));
% high=size(find(y_1==3));

[xtr_all_3, xte_all_3] = kfcv(x3,K,'off');
% [ytr_all_3, yte_all_3] = kfcv(y_1,K,'off');

[xtr_all_4, xte_all_4] = kfcv(x4,K,'off');
% [ytr_all_4, yte_all_4] = kfcv(y_1,K,'off');

% %case2_session1
% %基分类器数据输入准备
% a1=cell2mat(xtr_all_3(1,:));
% a2=cell2mat(xtr_all_3(2,:));
% a3=cell2mat(xtr_all_3(3,:));
% a4=cell2mat(xtr_all_3(4,:));
% a5=cell2mat(xtr_all_3(5,:));
% a6=cell2mat(xtr_all_3(6,:));
% 
% %输入数据附标签
% a1(:,71)=repmat(y,5,1);
% a2(:,71)=repmat(y,5,1);
% a3(:,71)=repmat(y,5,1);
% a4(:,71)=repmat(y,5,1);
% a5(:,71)=repmat(y,5,1);
% a6(:,71)=repmat(y,5,1);
% 
% %剩余5位被试分开
% a1_1=a1(1:1450,:); a1_2=a1(1451:2900,:); a1_3=a1(2901:4350,:); a1_4=a1(4351:5800,:); a1_5=a1(5801:7250,:);
% a2_1=a2(1:1450,:); a2_2=a2(1451:2900,:); a2_3=a2(2901:4350,:); a2_4=a2(4351:5800,:); a2_5=a2(5801:7250,:);
% a3_1=a3(1:1450,:); a3_2=a3(1451:2900,:); a3_3=a3(2901:4350,:); a3_4=a3(4351:5800,:); a3_5=a3(5801:7250,:);
% a4_1=a4(1:1450,:); a4_2=a4(1451:2900,:); a4_3=a4(2901:4350,:); a4_4=a4(4351:5800,:); a4_5=a4(5801:7250,:);
% a5_1=a5(1:1450,:); a5_2=a5(1451:2900,:); a5_3=a5(2901:4350,:); a5_4=a5(4351:5800,:); a5_5=a5(5801:7250,:);
% a6_1=a6(1:1450,:); a6_2=a6(1451:2900,:); a6_3=a6(2901:4350,:); a6_4=a6(4351:5800,:); a6_5=a6(5801:7250,:);
% 
% %分别取5位被试的训练集
% n=1400; idx=randperm(1450); idx=idx(1:n);
% BC1_1=a1_1(idx,:); BC1_2=a1_2(idx,:); BC1_3=a1_3(idx,:); BC1_4=a1_4(idx,:); BC1_5=a1_5(idx,:);
% BC2_1=a2_1(idx,:); BC2_2=a2_2(idx,:); BC2_3=a2_3(idx,:); BC2_4=a2_4(idx,:); BC2_5=a2_5(idx,:);
% BC3_1=a3_1(idx,:); BC3_2=a3_2(idx,:); BC3_3=a3_3(idx,:); BC3_4=a3_4(idx,:); BC3_5=a3_5(idx,:);
% BC4_1=a4_1(idx,:); BC4_2=a4_2(idx,:); BC4_3=a4_3(idx,:); BC4_4=a4_4(idx,:); BC4_5=a4_5(idx,:);
% BC5_1=a5_1(idx,:); BC5_2=a5_2(idx,:); BC5_3=a5_3(idx,:); BC5_4=a5_4(idx,:); BC5_5=a5_5(idx,:);
% BC6_1=a6_1(idx,:); BC6_2=a6_2(idx,:); BC6_3=a6_3(idx,:); BC6_4=a6_4(idx,:); BC6_5=a6_5(idx,:);
% 
% %5个基分类器训练集标签
% y_train_BC1_1=BC1_1(:,71);y_train_BC1_2=BC1_2(:,71);y_train_BC1_3=BC1_3(:,71);y_train_BC1_4=BC1_4(:,71);y_train_BC1_5=BC1_5(:,71);
% y_train_BC2_1=BC2_1(:,71);y_train_BC2_2=BC2_2(:,71);y_train_BC2_3=BC2_3(:,71);y_train_BC2_4=BC2_4(:,71);y_train_BC2_5=BC2_5(:,71);
% y_train_BC3_1=BC3_1(:,71);y_train_BC3_2=BC3_2(:,71);y_train_BC3_3=BC3_3(:,71);y_train_BC3_4=BC3_4(:,71);y_train_BC3_5=BC3_5(:,71);
% y_train_BC4_1=BC4_1(:,71);y_train_BC4_2=BC4_2(:,71);y_train_BC4_3=BC4_3(:,71);y_train_BC4_4=BC4_4(:,71);y_train_BC4_5=BC4_5(:,71);
% y_train_BC5_1=BC5_1(:,71);y_train_BC5_2=BC5_1(:,71);y_train_BC5_3=BC5_3(:,71);y_train_BC5_4=BC5_4(:,71);y_train_BC5_5=BC5_5(:,71);
% y_train_BC6_1=BC6_1(:,71);y_train_BC6_2=BC6_2(:,71);y_train_BC6_3=BC6_3(:,71);y_train_BC6_4=BC6_4(:,71);y_train_BC6_5=BC6_5(:,71);
% 
% %%%%
% %5个基分类器训练集
% BC1_1(:,71)=[]; BC1_2(:,71)=[]; BC1_3(:,71)=[]; BC1_4(:,71)=[]; BC1_5(:,71)=[];
% BC2_1(:,71)=[]; BC2_2(:,71)=[]; BC2_3(:,71)=[]; BC2_4(:,71)=[]; BC2_5(:,71)=[];
% BC3_1(:,71)=[]; BC3_2(:,71)=[]; BC3_3(:,71)=[]; BC3_4(:,71)=[]; BC3_5(:,71)=[];
% BC4_1(:,71)=[]; BC4_2(:,71)=[]; BC4_3(:,71)=[]; BC4_4(:,71)=[]; BC4_5(:,71)=[];
% BC5_1(:,71)=[]; BC5_2(:,71)=[]; BC5_3(:,71)=[]; BC5_4(:,71)=[]; BC5_5(:,71)=[];
% BC6_1(:,71)=[]; BC6_2(:,71)=[]; BC6_3(:,71)=[]; BC6_4(:,71)=[]; BC6_5(:,71)=[];
% 
% %除去训练集的测试集
% BC_test1_1=a1_1;BC_test1_1(idx,:)=[];BC_test1_2=a1_2;BC_test1_2(idx,:)=[];BC_test1_3=a1_3;BC_test1_3(idx,:)=[];BC_test1_4=a1_4;BC_test1_4(idx,:)=[];
% BC_test1_5=a1_5;BC_test1_5(idx,:)=[];
% 
% BC_test2_1=a2_1;BC_test2_1(idx,:)=[];BC_test2_2=a2_2;BC_test2_2(idx,:)=[];BC_test2_3=a2_3;BC_test2_3(idx,:)=[];BC_test2_4=a2_4;BC_test2_4(idx,:)=[];
% BC_test2_5=a2_5;BC_test2_5(idx,:)=[];
% 
% BC_test3_1=a3_1;BC_test3_1(idx,:)=[];BC_test3_2=a3_2;BC_test3_2(idx,:)=[];BC_test3_3=a3_3;BC_test3_3(idx,:)=[];BC_test3_4=a3_4;BC_test3_4(idx,:)=[];
% BC_test3_5=a3_5;BC_test3_5(idx,:)=[];
% 
% BC_test4_1=a4_1;BC_test4_1(idx,:)=[];BC_test4_2=a4_2;BC_test4_2(idx,:)=[];BC_test4_3=a4_3;BC_test4_3(idx,:)=[];BC_test4_4=a4_4;BC_test4_4(idx,:)=[];
% BC_test4_5=a4_5;BC_test4_5(idx,:)=[];
% 
% BC_test5_1=a5_1;BC_test5_1(idx,:)=[];BC_test5_2=a5_2;BC_test5_2(idx,:)=[];BC_test5_3=a5_3;BC_test5_3(idx,:)=[];BC_test5_4=a5_4;BC_test5_4(idx,:)=[];
% BC_test5_5=a5_5;BC_test5_5(idx,:)=[];
% 
% BC_test6_1=a6_1;BC_test6_1(idx,:)=[];BC_test6_2=a6_2;BC_test6_2(idx,:)=[];BC_test6_3=a6_3;BC_test6_3(idx,:)=[];BC_test6_4=a6_4;BC_test6_4(idx,:)=[];
% BC_test6_5=a6_5;BC_test6_5(idx,:)=[];
% 
% %5个基分类器测试集标签
% y_test_BC1_1=BC_test1_1(:,71);y_test_BC1_2=BC_test1_2(:,71);y_test_BC1_3=BC_test1_3(:,71);y_test_BC1_4=BC_test1_4(:,71);y_test_BC1_5=BC_test1_5(:,71);
% y_test_BC2_1=BC_test2_1(:,71);y_test_BC2_2=BC_test1_2(:,71);y_test_BC2_3=BC_test2_3(:,71);y_test_BC2_4=BC_test2_4(:,71);y_test_BC2_5=BC_test2_5(:,71);
% y_test_BC3_1=BC_test3_1(:,71);y_test_BC3_2=BC_test3_2(:,71);y_test_BC3_3=BC_test3_3(:,71);y_test_BC3_4=BC_test3_4(:,71);y_test_BC3_5=BC_test3_5(:,71);
% y_test_BC4_1=BC_test4_1(:,71);y_test_BC4_2=BC_test4_2(:,71);y_test_BC4_3=BC_test4_3(:,71);y_test_BC4_4=BC_test4_4(:,71);y_test_BC4_5=BC_test4_5(:,71);
% y_test_BC5_1=BC_test5_1(:,71);y_test_BC5_2=BC_test5_2(:,71);y_test_BC5_3=BC_test5_3(:,71);y_test_BC5_4=BC_test5_4(:,71);y_test_BC5_5=BC_test5_5(:,71);
% y_test_BC6_1=BC_test6_1(:,71);y_test_BC6_2=BC_test6_2(:,71);y_test_BC6_3=BC_test6_3(:,71);y_test_BC6_4=BC_test6_4(:,71);y_test_BC6_5=BC_test6_5(:,71);
% %5个基分类器测试集标签，按列排
% y_test_BC1=[y_test_BC1_1;y_test_BC1_2;y_test_BC1_3;y_test_BC1_4;y_test_BC1_5];
% y_test_BC2=[y_test_BC2_1;y_test_BC2_2;y_test_BC2_3;y_test_BC2_4;y_test_BC2_5];
% y_test_BC3=[y_test_BC3_1;y_test_BC3_2;y_test_BC3_3;y_test_BC3_4;y_test_BC3_5];
% y_test_BC4=[y_test_BC4_1;y_test_BC4_2;y_test_BC4_3;y_test_BC4_4;y_test_BC4_5];
% y_test_BC5=[y_test_BC5_1;y_test_BC5_2;y_test_BC5_3;y_test_BC5_4;y_test_BC5_5];
% y_test_BC6=[y_test_BC6_1;y_test_BC6_2;y_test_BC6_3;y_test_BC6_4;y_test_BC6_5];
% 
% %分别取5位被试的测试集
% BC_test1_1(:,71)=[]; BC_test1_2(:,71)=[]; BC_test1_3(:,71)=[]; BC_test1_4(:,71)=[]; BC_test1_5(:,71)=[];
% BC_test2_1(:,71)=[]; BC_test2_2(:,71)=[]; BC_test2_3(:,71)=[]; BC_test2_4(:,71)=[]; BC_test2_5(:,71)=[];
% BC_test3_1(:,71)=[]; BC_test3_2(:,71)=[]; BC_test3_3(:,71)=[]; BC_test3_4(:,71)=[]; BC_test3_5(:,71)=[];
% BC_test4_1(:,71)=[]; BC_test4_2(:,71)=[]; BC_test4_3(:,71)=[]; BC_test4_4(:,71)=[]; BC_test4_5(:,71)=[];
% BC_test5_1(:,71)=[]; BC_test5_2(:,71)=[]; BC_test5_3(:,71)=[]; BC_test5_4(:,71)=[]; BC_test5_5(:,71)=[];
% BC_test6_1(:,71)=[]; BC_test6_2(:,71)=[]; BC_test6_3(:,71)=[]; BC_test6_4(:,71)=[]; BC_test6_5(:,71)=[];
% 
% %基分类器的测试集，5个基分类器对同一数据测试，得出并序7组数据，然后通过Q-statistic，选取最佳的几排预测结果
% BC_test1=[BC_test1_1;BC_test1_2;BC_test1_3;BC_test1_4;BC_test1_5];
% BC_test2=[BC_test2_1;BC_test2_2;BC_test2_3;BC_test2_4;BC_test2_5];
% BC_test3=[BC_test3_1;BC_test3_2;BC_test3_3;BC_test3_4;BC_test3_5];
% BC_test4=[BC_test4_1;BC_test4_2;BC_test4_3;BC_test4_4;BC_test4_5];
% BC_test5=[BC_test5_1;BC_test5_2;BC_test5_3;BC_test5_4;BC_test5_5];
% BC_test6=[BC_test6_1;BC_test6_2;BC_test6_3;BC_test6_4;BC_test6_5];
% 
% %次级分类器测试集数据输入准备，给6个ELM测试
% b1=cell2mat(xte_all_3(1,:));
% b2=cell2mat(xte_all_3(2,:));
% b3=cell2mat(xte_all_3(3,:));
% b4=cell2mat(xte_all_3(4,:));
% b5=cell2mat(xte_all_3(5,:));
% b6=cell2mat(xte_all_3(6,:));
% 
% %次级分类器测试集，做跨被试，分别做8次测试，得出8组结果，然后平均，才为任务一阶段1的平均结果
% y_test_DC=y;  %每位被试的标签都一样
% 
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session1\BC_test_3 BC_test1 BC_test2 BC_test3 BC_test4 BC_test5 BC_test6...
%      y_test_BC1 y_test_BC2 y_test_BC3 y_test_BC4 y_test_BC5 y_test_BC6
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session1\BC_train_3 BC1_1 BC1_2 BC1_3 BC1_4 BC1_5...
%      BC2_1 BC2_2 BC2_3 BC2_4 BC2_5 BC3_1 BC3_2 BC3_3 BC3_4 BC3_5 BC4_1 BC4_2 BC4_3 BC4_4 BC4_5...
%      BC5_1 BC5_2 BC5_3 BC5_4 BC5_5 BC6_1 BC6_2 BC6_3 BC6_4 BC6_5...
%      y_train_BC1_1 y_train_BC1_2 y_train_BC1_3 y_train_BC1_4 y_train_BC1_5...
%      y_train_BC2_1 y_train_BC2_2 y_train_BC2_3 y_train_BC2_4 y_train_BC2_5...
%      y_train_BC3_1 y_train_BC3_2 y_train_BC3_3 y_train_BC3_4 y_train_BC3_5...
%      y_train_BC4_1 y_train_BC4_2 y_train_BC4_3 y_train_BC4_4 y_train_BC4_5...
%      y_train_BC5_1 y_train_BC5_2 y_train_BC5_3 y_train_BC5_4 y_train_BC5_5...
%      y_train_BC6_1 y_train_BC6_2 y_train_BC6_3 y_train_BC6_4 y_train_BC6_5
% 
%  save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session1\DC_test_3 b1 b2 b3 b4 b5 b6 y_test_DC

%case2_session2
%基分类器数据输入准备
a1=cell2mat(xtr_all_4(1,:));
a2=cell2mat(xtr_all_4(2,:));
a3=cell2mat(xtr_all_4(3,:));
a4=cell2mat(xtr_all_4(4,:));
a5=cell2mat(xtr_all_4(5,:));
a6=cell2mat(xtr_all_4(6,:));

%输入数据附标签
a1(:,71)=repmat(y,5,1);
a2(:,71)=repmat(y,5,1);
a3(:,71)=repmat(y,5,1);
a4(:,71)=repmat(y,5,1);
a5(:,71)=repmat(y,5,1);
a6(:,71)=repmat(y,5,1);

%剩余5位被试分开
a1_1=a1(1:1450,:); a1_2=a1(1451:2900,:); a1_3=a1(2901:4350,:); a1_4=a1(4351:5800,:); a1_5=a1(5801:7250,:);
a2_1=a2(1:1450,:); a2_2=a2(1451:2900,:); a2_3=a2(2901:4350,:); a2_4=a2(4351:5800,:); a2_5=a2(5801:7250,:);
a3_1=a3(1:1450,:); a3_2=a3(1451:2900,:); a3_3=a3(2901:4350,:); a3_4=a3(4351:5800,:); a3_5=a3(5801:7250,:);
a4_1=a4(1:1450,:); a4_2=a4(1451:2900,:); a4_3=a4(2901:4350,:); a4_4=a4(4351:5800,:); a4_5=a4(5801:7250,:);
a5_1=a5(1:1450,:); a5_2=a5(1451:2900,:); a5_3=a5(2901:4350,:); a5_4=a5(4351:5800,:); a5_5=a5(5801:7250,:);
a6_1=a6(1:1450,:); a6_2=a6(1451:2900,:); a6_3=a6(2901:4350,:); a6_4=a6(4351:5800,:); a6_5=a6(5801:7250,:);

%分别取5位被试的训练集
n=1400; idx=randperm(1450); idx=idx(1:n);
BC1_1=a1_1(idx,:); BC1_2=a1_2(idx,:); BC1_3=a1_3(idx,:); BC1_4=a1_4(idx,:); BC1_5=a1_5(idx,:);
BC2_1=a2_1(idx,:); BC2_2=a2_2(idx,:); BC2_3=a2_3(idx,:); BC2_4=a2_4(idx,:); BC2_5=a2_5(idx,:);
BC3_1=a3_1(idx,:); BC3_2=a3_2(idx,:); BC3_3=a3_3(idx,:); BC3_4=a3_4(idx,:); BC3_5=a3_5(idx,:);
BC4_1=a4_1(idx,:); BC4_2=a4_2(idx,:); BC4_3=a4_3(idx,:); BC4_4=a4_4(idx,:); BC4_5=a4_5(idx,:);
BC5_1=a5_1(idx,:); BC5_2=a5_2(idx,:); BC5_3=a5_3(idx,:); BC5_4=a5_4(idx,:); BC5_5=a5_5(idx,:);
BC6_1=a6_1(idx,:); BC6_2=a6_2(idx,:); BC6_3=a6_3(idx,:); BC6_4=a6_4(idx,:); BC6_5=a6_5(idx,:);

%5个基分类器训练集标签
y_train_BC1_1=BC1_1(:,71);y_train_BC1_2=BC1_2(:,71);y_train_BC1_3=BC1_3(:,71);y_train_BC1_4=BC1_4(:,71);y_train_BC1_5=BC1_5(:,71);
y_train_BC2_1=BC2_1(:,71);y_train_BC2_2=BC2_2(:,71);y_train_BC2_3=BC2_3(:,71);y_train_BC2_4=BC2_4(:,71);y_train_BC2_5=BC2_5(:,71);
y_train_BC3_1=BC3_1(:,71);y_train_BC3_2=BC3_2(:,71);y_train_BC3_3=BC3_3(:,71);y_train_BC3_4=BC3_4(:,71);y_train_BC3_5=BC3_5(:,71);
y_train_BC4_1=BC4_1(:,71);y_train_BC4_2=BC4_2(:,71);y_train_BC4_3=BC4_3(:,71);y_train_BC4_4=BC4_4(:,71);y_train_BC4_5=BC4_5(:,71);
y_train_BC5_1=BC5_1(:,71);y_train_BC5_2=BC5_1(:,71);y_train_BC5_3=BC5_3(:,71);y_train_BC5_4=BC5_4(:,71);y_train_BC5_5=BC5_5(:,71);
y_train_BC6_1=BC6_1(:,71);y_train_BC6_2=BC6_2(:,71);y_train_BC6_3=BC6_3(:,71);y_train_BC6_4=BC6_4(:,71);y_train_BC6_5=BC6_5(:,71);

%%%%
%5个基分类器训练集
BC1_1(:,71)=[]; BC1_2(:,71)=[]; BC1_3(:,71)=[]; BC1_4(:,71)=[]; BC1_5(:,71)=[];
BC2_1(:,71)=[]; BC2_2(:,71)=[]; BC2_3(:,71)=[]; BC2_4(:,71)=[]; BC2_5(:,71)=[];
BC3_1(:,71)=[]; BC3_2(:,71)=[]; BC3_3(:,71)=[]; BC3_4(:,71)=[]; BC3_5(:,71)=[];
BC4_1(:,71)=[]; BC4_2(:,71)=[]; BC4_3(:,71)=[]; BC4_4(:,71)=[]; BC4_5(:,71)=[];
BC5_1(:,71)=[]; BC5_2(:,71)=[]; BC5_3(:,71)=[]; BC5_4(:,71)=[]; BC5_5(:,71)=[];
BC6_1(:,71)=[]; BC6_2(:,71)=[]; BC6_3(:,71)=[]; BC6_4(:,71)=[]; BC6_5(:,71)=[];

%除去训练集的测试集
BC_test1_1=a1_1;BC_test1_1(idx,:)=[];BC_test1_2=a1_2;BC_test1_2(idx,:)=[];BC_test1_3=a1_3;BC_test1_3(idx,:)=[];BC_test1_4=a1_4;BC_test1_4(idx,:)=[];
BC_test1_5=a1_5;BC_test1_5(idx,:)=[];

BC_test2_1=a2_1;BC_test2_1(idx,:)=[];BC_test2_2=a2_2;BC_test2_2(idx,:)=[];BC_test2_3=a2_3;BC_test2_3(idx,:)=[];BC_test2_4=a2_4;BC_test2_4(idx,:)=[];
BC_test2_5=a2_5;BC_test2_5(idx,:)=[];

BC_test3_1=a3_1;BC_test3_1(idx,:)=[];BC_test3_2=a3_2;BC_test3_2(idx,:)=[];BC_test3_3=a3_3;BC_test3_3(idx,:)=[];BC_test3_4=a3_4;BC_test3_4(idx,:)=[];
BC_test3_5=a3_5;BC_test3_5(idx,:)=[];

BC_test4_1=a4_1;BC_test4_1(idx,:)=[];BC_test4_2=a4_2;BC_test4_2(idx,:)=[];BC_test4_3=a4_3;BC_test4_3(idx,:)=[];BC_test4_4=a4_4;BC_test4_4(idx,:)=[];
BC_test4_5=a4_5;BC_test4_5(idx,:)=[];

BC_test5_1=a5_1;BC_test5_1(idx,:)=[];BC_test5_2=a5_2;BC_test5_2(idx,:)=[];BC_test5_3=a5_3;BC_test5_3(idx,:)=[];BC_test5_4=a5_4;BC_test5_4(idx,:)=[];
BC_test5_5=a5_5;BC_test5_5(idx,:)=[];

BC_test6_1=a6_1;BC_test6_1(idx,:)=[];BC_test6_2=a6_2;BC_test6_2(idx,:)=[];BC_test6_3=a6_3;BC_test6_3(idx,:)=[];BC_test6_4=a6_4;BC_test6_4(idx,:)=[];
BC_test6_5=a6_5;BC_test6_5(idx,:)=[];

%5个基分类器测试集标签
y_test_BC1_1=BC_test1_1(:,71);y_test_BC1_2=BC_test1_2(:,71);y_test_BC1_3=BC_test1_3(:,71);y_test_BC1_4=BC_test1_4(:,71);y_test_BC1_5=BC_test1_5(:,71);
y_test_BC2_1=BC_test2_1(:,71);y_test_BC2_2=BC_test1_2(:,71);y_test_BC2_3=BC_test2_3(:,71);y_test_BC2_4=BC_test2_4(:,71);y_test_BC2_5=BC_test2_5(:,71);
y_test_BC3_1=BC_test3_1(:,71);y_test_BC3_2=BC_test3_2(:,71);y_test_BC3_3=BC_test3_3(:,71);y_test_BC3_4=BC_test3_4(:,71);y_test_BC3_5=BC_test3_5(:,71);
y_test_BC4_1=BC_test4_1(:,71);y_test_BC4_2=BC_test4_2(:,71);y_test_BC4_3=BC_test4_3(:,71);y_test_BC4_4=BC_test4_4(:,71);y_test_BC4_5=BC_test4_5(:,71);
y_test_BC5_1=BC_test5_1(:,71);y_test_BC5_2=BC_test5_2(:,71);y_test_BC5_3=BC_test5_3(:,71);y_test_BC5_4=BC_test5_4(:,71);y_test_BC5_5=BC_test5_5(:,71);
y_test_BC6_1=BC_test6_1(:,71);y_test_BC6_2=BC_test6_2(:,71);y_test_BC6_3=BC_test6_3(:,71);y_test_BC6_4=BC_test6_4(:,71);y_test_BC6_5=BC_test6_5(:,71);
%5个基分类器测试集标签，按列排
y_test_BC1=[y_test_BC1_1;y_test_BC1_2;y_test_BC1_3;y_test_BC1_4;y_test_BC1_5];
y_test_BC2=[y_test_BC2_1;y_test_BC2_2;y_test_BC2_3;y_test_BC2_4;y_test_BC2_5];
y_test_BC3=[y_test_BC3_1;y_test_BC3_2;y_test_BC3_3;y_test_BC3_4;y_test_BC3_5];
y_test_BC4=[y_test_BC4_1;y_test_BC4_2;y_test_BC4_3;y_test_BC4_4;y_test_BC4_5];
y_test_BC5=[y_test_BC5_1;y_test_BC5_2;y_test_BC5_3;y_test_BC5_4;y_test_BC5_5];
y_test_BC6=[y_test_BC6_1;y_test_BC6_2;y_test_BC6_3;y_test_BC6_4;y_test_BC6_5];

%分别取5位被试的测试集
BC_test1_1(:,71)=[]; BC_test1_2(:,71)=[]; BC_test1_3(:,71)=[]; BC_test1_4(:,71)=[]; BC_test1_5(:,71)=[];
BC_test2_1(:,71)=[]; BC_test2_2(:,71)=[]; BC_test2_3(:,71)=[]; BC_test2_4(:,71)=[]; BC_test2_5(:,71)=[];
BC_test3_1(:,71)=[]; BC_test3_2(:,71)=[]; BC_test3_3(:,71)=[]; BC_test3_4(:,71)=[]; BC_test3_5(:,71)=[];
BC_test4_1(:,71)=[]; BC_test4_2(:,71)=[]; BC_test4_3(:,71)=[]; BC_test4_4(:,71)=[]; BC_test4_5(:,71)=[];
BC_test5_1(:,71)=[]; BC_test5_2(:,71)=[]; BC_test5_3(:,71)=[]; BC_test5_4(:,71)=[]; BC_test5_5(:,71)=[];
BC_test6_1(:,71)=[]; BC_test6_2(:,71)=[]; BC_test6_3(:,71)=[]; BC_test6_4(:,71)=[]; BC_test6_5(:,71)=[];

%基分类器的测试集，5个基分类器对同一数据测试，得出并序7组数据，然后通过Q-statistic，选取最佳的几排预测结果
BC_test1=[BC_test1_1;BC_test1_2;BC_test1_3;BC_test1_4;BC_test1_5];
BC_test2=[BC_test2_1;BC_test2_2;BC_test2_3;BC_test2_4;BC_test2_5];
BC_test3=[BC_test3_1;BC_test3_2;BC_test3_3;BC_test3_4;BC_test3_5];
BC_test4=[BC_test4_1;BC_test4_2;BC_test4_3;BC_test4_4;BC_test4_5];
BC_test5=[BC_test5_1;BC_test5_2;BC_test5_3;BC_test5_4;BC_test5_5];
BC_test6=[BC_test6_1;BC_test6_2;BC_test6_3;BC_test6_4;BC_test6_5];

%次级分类器测试集数据输入准备，给6个ELM测试
b1=cell2mat(xte_all_4(1,:));
b2=cell2mat(xte_all_4(2,:));
b3=cell2mat(xte_all_4(3,:));
b4=cell2mat(xte_all_4(4,:));
b5=cell2mat(xte_all_4(5,:));
b6=cell2mat(xte_all_4(6,:));

%次级分类器测试集，做跨被试，分别做8次测试，得出8组结果，然后平均，才为任务一阶段1的平均结果
y_test_DC=y;  %每位被试的标签都一样

save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session2\BC_test_4 BC_test1 BC_test2 BC_test3 BC_test4 BC_test5 BC_test6...
     y_test_BC1 y_test_BC2 y_test_BC3 y_test_BC4 y_test_BC5 y_test_BC6
save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session2\BC_train_4 BC1_1 BC1_2 BC1_3 BC1_4 BC1_5...
     BC2_1 BC2_2 BC2_3 BC2_4 BC2_5 BC3_1 BC3_2 BC3_3 BC3_4 BC3_5 BC4_1 BC4_2 BC4_3 BC4_4 BC4_5...
     BC5_1 BC5_2 BC5_3 BC5_4 BC5_5 BC6_1 BC6_2 BC6_3 BC6_4 BC6_5...
     y_train_BC1_1 y_train_BC1_2 y_train_BC1_3 y_train_BC1_4 y_train_BC1_5...
     y_train_BC2_1 y_train_BC2_2 y_train_BC2_3 y_train_BC2_4 y_train_BC2_5...
     y_train_BC3_1 y_train_BC3_2 y_train_BC3_3 y_train_BC3_4 y_train_BC3_5...
     y_train_BC4_1 y_train_BC4_2 y_train_BC4_3 y_train_BC4_4 y_train_BC4_5...
     y_train_BC5_1 y_train_BC5_2 y_train_BC5_3 y_train_BC5_4 y_train_BC5_5...
     y_train_BC6_1 y_train_BC6_2 y_train_BC6_3 y_train_BC6_4 y_train_BC6_5

 save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\data\case2_session2\DC_test_4 b1 b2 b3 b4 b5 b6 y_test_DC



