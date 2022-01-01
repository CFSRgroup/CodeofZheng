% %本程序用于输入确定和目标类别分析
clc;
clear;
close all;

for pp=1:6
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_1'])
x1=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_2'])
x2=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_3'])
x3=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_4'])
x4=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_5'])
x5=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_6'])
x6=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_7'])
x7=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_8'])
x8=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_9'])
x9=x;
eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session1\s' num2str(pp) '_10'])
x10=x;

xx=[x1;x2;x3;x4;x5;x6;x7;x8;x9;x10];
c=isnan(xx);
max(max(c))
xx(isnan(xx))=0;

x=zscore(xx);

y1=1*ones(size(x1,1),1);
y2=2*ones(size(x1,1),1);
y3=3*ones(size(x1,1),1);
y=[y1;y2;y3;y1;y2;y3;y1;y2;y3;y1];
eval(['save F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(pp) '_1 x y'])
end

% for pp=1:6
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_1'])
% x1=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_2'])
% x2=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_3'])
% x3=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_4'])
% x4=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_5'])
% x5=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_6'])
% x6=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_7'])
% x7=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_8'])
% x8=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_9'])
% x9=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex2_feature\session2\s' num2str(pp) '_10'])
% x10=x;
% x=zscore([x1;x2;x3;x4;x5;x6;x7;x8;x9;x10]);
% y1=1*ones(size(x1,1),1);
% y2=2*ones(size(x1,1),1);
% y3=3*ones(size(x1,1),1);
% y=[y1;y2;y3;y1;y2;y3;y1;y2;y3;y1];
% eval(['save F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(pp) '_2 x y'])
% end