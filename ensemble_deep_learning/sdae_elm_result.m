clc;
clear;
close all;
warning off;

% %case1_session1
% %A
% matrix_A=[469 154 100;257 502 178;174 244 622];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_A); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_A=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_A=8.0625;
% 
% %B
% matrix_B=[677 216 104;122 440 315;101 244 481];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_B); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_B=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_B=7.75;
% 
% %C
% matrix_C=[411 38 87;451 822 313;38 40 500];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_C); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_C=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_C=7.7656;
% 
% %D
% matrix_D=[263 167 154;9 400 11;628 333 735];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_D); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_D=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_D=7.7031;
% 
% %E
% matrix_E=[509 10 143;11 607 12;380 283 745];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_E); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_E=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_E=7.3750;
% 
% %F
% matrix_F=[452 7 329;7 459 12;441 434 559];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_F); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_F=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_F=7.3906;
% 
% %G
% matrix_G=[700 346 443;190 533 12;10 21 445];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_G); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_G=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_G=7.7344;
% 
% %H
% matrix_H=[542 176 267;265 700 152;93 24 481];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_H); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_H=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_H=7.1563;
% 
% case1_session1_per=[per_index_A;per_index_B;per_index_C;per_index_D;per_index_E;per_index_F;per_index_G;per_index_H];
% case1_session1_time=[time_A;time_B;time_C;time_D;time_E;time_F;time_G;time_H];
% 
% case1_session1_time_sum=sum(case1_session1_time);
% case1_session1_time_sd=std(case1_session1_time);
% case1_session1_Time=[case1_session1_time_sum case1_session1_time_sd];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case1_session1_result case1_session1_per case1_session1_time case1_session1_Time


% %case1_session2
% %A
% matrix_A=[239 99 120;534 735 67;127 66 713];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_A); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_A=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_A=7.4531;
% 
% %B
% matrix_B=[527 92 117;155 468 228;218 340 555];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_B); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_B=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_B=7.7188;
% 
% %C
% matrix_C=[313 243 189;6 650 9;581 7 702];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_C); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_C=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_C=7.6719;
% 
% %D
% matrix_D=[471 3 2;423 467 377;6 430 521];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_D); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_D=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_D=7.8906;
% 
% %E
% matrix_E=[658 268 325;233 629 10;9 3 565];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_E); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_E=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_E=7.2969;
% 
% %F
% matrix_F=[426 266 223;166 501 93;308 133 584];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_F); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_F=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_F=7.5469;
% 
% %G
% matrix_G=[359 225 30;491 633 159;50 42 711];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_G); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_G=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_G=8.2656;
% 
% %H
% matrix_H=[669 15 21;23 754 479;208 131 400];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_H); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_H=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_H=7.6799;
% 
% case1_session2_per=[per_index_A;per_index_B;per_index_C;per_index_D;per_index_E;per_index_F;per_index_G;per_index_H];
% case1_session2_time=[time_A;time_B;time_C;time_D;time_E;time_F;time_G;time_H];
% 
% case1_session2_time_sum=sum(case1_session2_time);
% case1_session2_time_sd=std(case1_session2_time);
% case1_session2_Time=[case1_session2_time_sum case1_session2_time_sd];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case1_session2_result case1_session2_per case1_session2_time case1_session2_Time

% %case2_session1
% %I
% matrix_I=[526 313 151;2 120 5;52 2 279];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_I); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_I=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_I=3.4531;
% 
% %J
% matrix_J=[489 322 228;5 109 7;86 4 200];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_J); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_J=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_J=3.6406;
% 
% %K
% matrix_K=[400 60 17;110 335 96;70 40 322];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_K); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_K=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_K=3.7188;
% 
% %L
% matrix_L=[397 52 62;14 300 44;169 83 329];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_L); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_L=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_L=3.5781;
% 
% %M
% matrix_M=[452 27 43;87 297 178;41 111 214];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_M); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_M=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_M=3.7656;
% 
% %N
% matrix_N=[300 145 24;109 211 61;171 79 350];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_N); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_N=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_N=3.8906;
% 
% 
% case2_session1_per=[per_index_I;per_index_J;per_index_K;per_index_L;per_index_M;per_index_N];
% case2_session1_time=[time_I;time_J;time_K;time_L;time_M;time_N];
% 
% case2_session1_time_sum=sum(case2_session1_time);
% case2_session1_time_sd=std(case2_session1_time);
% case2_session1_Time=[case2_session1_time_sum case2_session1_time_sd];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case2_session1_result case2_session1_per case2_session1_time case2_session1_Time

% %case2_session2
% %I
% matrix_I=[400 144 45;104 240 90;76 51 300];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_I); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_I=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_I=3.5313;
% 
% %J
% matrix_J=[371 40 45;55 263 202;154 132 188];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_J); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_J=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_J=3.8438;
% 
% %K
% matrix_K=[470 14 138;20 320 19;90 101 278];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_K); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_K=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_K=3.3281;
% 
% %L
% matrix_L=[435 19 75;86 376 127;59 40 233];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_L); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_L=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_L=3.6563;
% 
% %M
% matrix_M=[436 16 10;42 258 171;102 161 254];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_M); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_M=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_M=3.8750;
% 
% %N
% matrix_N=[498 12 12;69 249 282;13 174 141];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_N); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_N=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% time_N=3.6573;
% 
% 
% case2_session2_per=[per_index_I;per_index_J;per_index_K;per_index_L;per_index_M;per_index_N];
% case2_session2_time=[time_I;time_J;time_K;time_L;time_M;time_N];
% 
% case2_session2_time_sum=sum(case2_session2_time);
% case2_session2_time_sd=std(case2_session2_time);
% case2_session2_Time=[case2_session2_time_sum case2_session2_time_sd];
% save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case2_session2_result case2_session2_per case2_session2_time case2_session2_Time



%上面数据处理完后，开始分析和总结，两组任务，两个阶段
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case1_session1_result
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case1_session2_result
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case2_session1_result
load F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case2_session2_result

%case1 
case1_per_single_mean=(case1_session1_per+case1_session2_per)./2;  %单个被试指标分析
case1_time=(case1_session1_Time+case1_session2_Time)./2;    %填表格，平均值和方差
case1_per_mean=(mean(case1_session1_per)+mean(case1_session2_per))./2;  %误差棒图平均值
case1_per_sd=(std(case1_session1_per)+std(case1_session2_per))./2;    %误差棒图误差

save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case1_result case1_per_single_mean case1_time case1_per_mean case1_per_sd

%case2
case2_per_single_mean=(case2_session1_per+case2_session2_per)./2;  %单个被试指标分析
case2_time=(case2_session1_Time+case2_session2_Time)./2;    %填表格，平均值和方差
case2_per_mean=(mean(case2_session1_per)+mean(case2_session2_per))./2;  %误差棒图平均值
case2_per_sd=(std(case2_session1_per)+std(case2_session2_per))./2;    %误差棒图误差

save F:\matlab\trial_procedure\study_1\ensemble_deep_learning\result\case2_result case2_per_single_mean case2_time case2_per_mean case2_per_sd
















