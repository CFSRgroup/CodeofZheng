%绘制3-D散点图
%author:郑展鹏
%2019/4/12
%note:任务1选择被试A_1，低中高负荷各取40个样例(共120样例)，分两阶段(#1-#3;#4-#6)，各阶段取20个样例，低中高样例均连续。选择通道F3，(#1-#3)
%note:任务2选择被试J_1，低中高负荷各取40个样例(共120样例)，分两阶段(#1-#3;#4-#6)，各阶段取20个样例，低中高样例均连续。选择通道F3，(#1-#3)

clc;
clear;
close all;

load('F:\matlab\trial_procedure\study_1\features\ex1\s1_1.mat')
scatterplot_case1_A([1:40;41:80;81:120],:)=x([1:40;451:490;901:940],1:3);
scatterplot_case1_A=scatterplot_case1_A';
save F:\matlab\trial_procedure\study_1\baseline_performance\scatterplot\scatterplot_case1_A scatterplot_case1_A

load('F:\matlab\trial_procedure\study_1\features\ex2\s2_1.mat')
scatterplot_case2_I([1:40;41:80;81:120],:)=x([1:40;146:185;291:330],1:3);
scatterplot_case2_I=scatterplot_case2_I';
save F:\matlab\trial_procedure\study_1\baseline_performance\scatterplot\scatterplot_case2_I scatterplot_case2_I

subplot(1,2,1);
S = 40*ones(1,40);
scatter3(scatterplot_case1_A(1,1:40),scatterplot_case1_A(2,1:40),scatterplot_case1_A(3,1:40),S,'go','filled');
hold on;
scatter3(scatterplot_case1_A(1,41:80),scatterplot_case1_A(2,41:80),scatterplot_case1_A(3,41:80),S,'b^','filled');
hold on;
scatter3(scatterplot_case1_A(1,81:120),scatterplot_case1_A(2,81:120),scatterplot_case1_A(3,81:120),S,'rs','filled');
hold off;
view(-40,30);
legend('Low MW instances in Case1','Medium MW instances in Case1','High MW instances in Case1');
title('(e) 3D-scatterplot for selected instances in Case1','FontWeight','bold');
grid on;

subplot(1,2,2);
S = 40*ones(1,40);
scatter3(scatterplot_case2_I(1,1:40),scatterplot_case2_I(2,1:40),scatterplot_case2_I(3,1:40),S,'go','filled');
hold on;
scatter3(scatterplot_case2_I(1,41:80),scatterplot_case2_I(2,41:80),scatterplot_case2_I(3,41:80),S,'b^','filled');
hold on;
scatter3(scatterplot_case2_I(1,81:120),scatterplot_case2_I(2,81:120),scatterplot_case2_I(3,81:120),S,'rs','filled');
hold off;
view(-40,30);
legend('Low MW instances in Case2','Medium MW instances in Case2','High MW instances in Case2');
title('(f) 3D-scatterplot for selected instances in Case2','FontWeight','bold');
grid on;





