% %��������������ȷ����Ŀ��������
% clc;
% clear;
% close all;
% 
% for pp=1:8
% eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session1\s' num2str(pp) '_1'])
% x1=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session1\s' num2str(pp) '_2'])
% x2=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session1\s' num2str(pp) '_3'])
% x3=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session1\s' num2str(pp) '_4'])
% x4=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session1\s' num2str(pp) '_5'])
% x5=x;
% eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session1\s' num2str(pp) '_6'])
% x6=x;
% x=zscore([x1;x2;x3;x4;x5;x6]); %�����ݷ���֮ǰ������ͨ����Ҫ�Ƚ����ݱ�׼����normalization�������ñ�׼��������ݽ������ݷ�����z=(x-mean(x))./std(x)
% y1=1*ones(size(x1,1),1); %low
% y2=2*ones(size(x1,1),1); %median
% y3=3*ones(size(x1,1),1); %high
% y=[y1;y2;y3;y3;y2;y1]; 
% eval(['save F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_1 x y'])
% end

for pp=1:8
eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session2\s' num2str(pp) '_1'])
x1=x;
eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session2\s' num2str(pp) '_2'])
x2=x;
eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session2\s' num2str(pp) '_3'])
x3=x;
eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session2\s' num2str(pp) '_4'])
x4=x;
eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session2\s' num2str(pp) '_5'])
x5=x;
eval(['load F:\matlab\trial_procedure\study_1\ex1_feature\session2\s' num2str(pp) '_6'])
x6=x;
x=zscore([x1;x2;x3;x4;x5;x6]); %�����ݷ���֮ǰ������ͨ����Ҫ�Ƚ����ݱ�׼����normalization�������ñ�׼��������ݽ������ݷ�����z=(x-mean(x))./std(x)
y1=1*ones(size(x1,1),1); %low
y2=2*ones(size(x1,1),1); %median
y3=3*ones(size(x1,1),1); %high
y=[y1;y2;y3;y3;y2;y1]; 
eval(['save F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_2 x y'])
end