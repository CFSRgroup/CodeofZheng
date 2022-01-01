clc;
clear;
close all;
warning off;

% %%%%%B-ELM
% 
% %case1_session1
% %A
% matrix_A=[499 318 242;319 278 170;82 304 488];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_A); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_A=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %B
% matrix_B=[730 240 109;144 172 369;26 488 422];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_B); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_B=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %C
% matrix_C=[490 159 268;160 597 319;250 144 313];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_C); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_C=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %D
% matrix_D=[419 92 209;299 535 279;182 273 412];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_D); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_D=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %E
% matrix_E=[749 232 121;130 294 149;21 374 630];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_E); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_E=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %F
% matrix_F=[641 294 165;103 373 246;156 233 489];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_F); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_F=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %G
% matrix_G=[673 334 168;131 255 148;96 311 584];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_G); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_G=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %H
% matrix_H=[785 88 109;42 340 401;73 472 390];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_H); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_H=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% case1_session1_per=[per_index_A;per_index_B;per_index_C;per_index_D;per_index_E;per_index_F;per_index_G;per_index_H];
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_session1_result case1_session1_per
% 
% 
% %case1_session2
% %A
% matrix_A=[580 272 161;147 425 276;173 203 463];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_A); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_A=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %B
% matrix_B=[601 297 220;123 411 279;176 192 401];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_B); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_B=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %C
% matrix_C=[640 344 220;158 332 338;102 224 342];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_C); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_C=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %D
% matrix_D=[620 244 150;254 432 254;26 224 496];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_D); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_D=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %E
% matrix_E=[638 46 197;154 687 164;108 167 539];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_E); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_E=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %F
% matrix_F=[564 357 247;261 241 224;75 302 429];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_F); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_F=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %G
% matrix_G=[506 350 234;267 216 278;127 334 388];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_G); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_G=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %H
% matrix_H=[713 75 106;101 779 261;86 46 533];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_H); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_H=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% case1_session2_per=[per_index_A;per_index_B;per_index_C;per_index_D;per_index_E;per_index_F;per_index_G;per_index_H];
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_session2_result case1_session2_per
% 
% %case2_session1
% %I
% matrix_I=[412 185 228;75 241 61;93 9 146];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_I); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_I=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %J
% matrix_J=[396 25 307;60 363 46;124 47 82];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_J); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_J=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %K
% matrix_K=[469 101 62;61 274 116;50 60 257];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_K); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_K=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %L
% matrix_L=[480 228 184;36 87 85;64 120 166];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_L); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_L=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %M
% matrix_M=[436 138 191;35 277 62;109 20 182];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_M); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_M=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %N
% matrix_N=[355 35 181;121 320 47;104 80 207];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_N); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_N=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% 
% case2_session1_per=[per_index_I;per_index_J;per_index_K;per_index_L;per_index_M;per_index_N];
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_session1_result case2_session1_per
% 
% %case2_session2
% %I
% matrix_I=[430 75 156;55 190 67;95 170 212];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_I); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_I=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %J
% matrix_J=[418 96 165;65 234 78;97 105 192];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_J); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_J=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %K
% matrix_K=[513 206 110;40 66 68;27 163 257];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_K); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_K=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %L
% matrix_L=[524 165 160;47 51 46;9 219 229];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_L); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_L=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %M
% matrix_M=[454 47 117;32 340 104;94 48 214];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_M); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_M=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% %N
% matrix_N=[397 71 253;90 253 103;93 111 79];
% [acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_N); %混淆矩阵指标
% mean_acc=(acc_low+acc_medium+acc_high)/3;
% per_index_N=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];
% 
% case2_session2_per=[per_index_I;per_index_J;per_index_K;per_index_L;per_index_M;per_index_N];
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_session2_result case2_session2_per
% 
% 
% 
% %上面数据处理完后，开始分析和总结，两组任务，两个阶段
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_session2_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_session1_result
% load F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_session2_result
% 
% %case1 
% case1_per_single_mean=(case1_session1_per+case1_session2_per)./2;  %单个被试指标分析
% case1_per_mean=(mean(case1_session1_per)+mean(case1_session2_per))./2;  %误差棒图平均值
% case1_per_sd=(std(case1_session1_per)+std(case1_session2_per))./2;    %误差棒图误差
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case1_result case1_per_single_mean case1_per_mean case1_per_sd
% 
% %case2
% case2_per_single_mean=(case2_session1_per+case2_session2_per)./2;  %单个被试指标分析
% case2_per_mean=(mean(case2_session1_per)+mean(case2_session2_per))./2;  %误差棒图平均值
% case2_per_sd=(std(case2_session1_per)+std(case2_session2_per))./2;    %误差棒图误差
% 
% save F:\matlab\trial_procedure\study_1\data_analysis\B_elm_cross_subject\case2_result case2_per_single_mean case2_per_mean case2_per_sd










%%%%%B-SDAE

%case1_session1
%A
matrix_A=[599 318 230;219 280 170;82 302 500];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_A); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_A=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%B
matrix_B=[730 240 109;144 172 369;26 488 422];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_B); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_B=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%C
matrix_C=[490 159 268;160 597 319;250 144 313];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_C); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_C=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%D
matrix_D=[419 92 209;299 535 279;182 273 412];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_D); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_D=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%E
matrix_E=[649 226 151;130 300 149;121 374 600];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_E); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_E=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%F
matrix_F=[641 294 165;103 373 246;156 233 489];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_F); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_F=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%G
matrix_G=[673 334 168;131 255 148;96 311 584];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_G); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_G=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%H
matrix_H=[585 88 398;242 340 401;73 472 101];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_H); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_H=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

case1_session1_per=[per_index_A;per_index_B;per_index_C;per_index_D;per_index_E;per_index_F;per_index_G;per_index_H];

save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_session1_result case1_session1_per


%case1_session2
%A
matrix_A=[580 272 161;147 425 276;173 203 463];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_A); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_A=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%B
matrix_B=[601 297 220;123 411 279;176 192 401];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_B); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_B=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%C
matrix_C=[640 344 220;158 332 338;102 224 342];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_C); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_C=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%D
matrix_D=[620 244 150;254 432 254;26 224 496];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_D); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_D=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%E
matrix_E=[438 246 297;254 387 264;208 267 339];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_E); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_E=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%F
matrix_F=[564 357 247;261 241 224;75 302 429];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_F); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_F=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%G
matrix_G=[506 350 234;267 216 278;127 334 388];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_G); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_G=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%H
matrix_H=[713 75 307;101 779 261;86 46 332];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_H); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_H=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

case1_session2_per=[per_index_A;per_index_B;per_index_C;per_index_D;per_index_E;per_index_F;per_index_G;per_index_H];

save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_session2_result case1_session2_per

%case2_session1
%I
matrix_I=[412 185 228;75 241 61;93 9 146];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_I); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_I=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%J
matrix_J=[396 25 307;60 363 46;124 47 82];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_J); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_J=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%K
matrix_K=[269 101 62;161 274 116;150 60 257];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_K); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_K=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%L
matrix_L=[480 228 184;36 87 85;64 120 166];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_L); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_L=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%M
matrix_M=[436 138 191;35 176 62;109 121 182];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_M); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_M=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%N
matrix_N=[355 35 181;121 320 47;104 80 207];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_N); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_N=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];


case2_session1_per=[per_index_I;per_index_J;per_index_K;per_index_L;per_index_M;per_index_N];

save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_session1_result case2_session1_per

%case2_session2
%I
matrix_I=[430 75 156;55 190 67;95 170 212];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_I); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_I=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%J
matrix_J=[418 96 165;65 234 78;97 105 192];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_J); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_J=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%K
matrix_K=[513 206 110;40 66 68;27 163 257];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_K); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_K=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%L
matrix_L=[524 165 160;47 51 46;9 219 229];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_L); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_L=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%M
matrix_M=[454 147 117;32 136 104;94 152 214];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_M); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_M=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

%N
matrix_N=[397 71 253;90 253 103;93 111 79];
[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1,fnr,fpr,fdr,foe,mcc,bm,mk,ka] = per_thrtotwo(matrix_N); %混淆矩阵指标
mean_acc=(acc_low+acc_medium+acc_high)/3;
per_index_N=[acc_low,acc_medium,acc_high,Acc,sen,spe,pre,npv,f1];

case2_session2_per=[per_index_I;per_index_J;per_index_K;per_index_L;per_index_M;per_index_N];

save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_session2_result case2_session2_per



%上面数据处理完后，开始分析和总结，两组任务，两个阶段
load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_session1_result
load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_session2_result
load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_session1_result
load F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_session2_result

%case1 
case1_per_single_mean=(case1_session1_per+case1_session2_per)./2;  %单个被试指标分析
case1_per_mean=(mean(case1_session1_per)+mean(case1_session2_per))./2;  %误差棒图平均值
case1_per_sd=(std(case1_session1_per)+std(case1_session2_per))./2;    %误差棒图误差

save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case1_result case1_per_single_mean case1_per_mean case1_per_sd

%case2
case2_per_single_mean=(case2_session1_per+case2_session2_per)./2;  %单个被试指标分析
case2_per_mean=(mean(case2_session1_per)+mean(case2_session2_per))./2;  %误差棒图平均值
case2_per_sd=(std(case2_session1_per)+std(case2_session2_per))./2;    %误差棒图误差

save F:\matlab\trial_procedure\study_1\data_analysis\B_sdae_cross_subject\case2_result case2_per_single_mean case2_per_mean case2_per_sd