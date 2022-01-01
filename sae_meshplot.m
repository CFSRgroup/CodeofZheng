clc;
clear;
close all;
warning off;

for pp=1:20
eval(['load F:\matlab\trial_procedure\study_1\data_analysis\sae_plot\task1_session1\h' num2str(pp)])
x1(pp,:)=plot_TestingAccuracy;
end

for pp=1:20
eval(['load F:\matlab\trial_procedure\study_1\data_analysis\sae_plot\task1_session2\h' num2str(pp)])
x2(pp,:)=plot_TestingAccuracy;
end

for pp=1:20
eval(['load F:\matlab\trial_procedure\study_1\data_analysis\sae_plot\task2_session1\h' num2str(pp)])
x3(pp,:)=plot_TestingAccuracy_1;
end

for pp=1:20
eval(['load F:\matlab\trial_procedure\study_1\data_analysis\sae_plot\task2_session2\h' num2str(pp)])
x4(pp,:)=plot_TestingAccuracy_1;
end

x11=(x1+x2)./2;
x22=(x3+x4)./2;

save F:\matlab\trial_procedure\study_1\data_analysis\sae_plot\data x11 x22

subplot(2,1,1);
mesh(x11);
title('(i) SAE: Case 1','FontWeight','bold');
% set(gca,'XTick',0:25:100);
% set(gca,'XTickLabel',{'0','25','50','75','100'});
% set(gca,'YTick',0:25:100);
% set(gca,'YTickLabel',{'0','25','50','75','100'});
xlabel('Number of neurons in layer 2','FontWeight','bold');
ylabel('Number of neurons in layer 1','FontWeight','bold');
zlabel('Accuracy','FontWeight','bold');


subplot(2,1,2);
mesh(x22);
title('(j) SAE: Case 2','FontWeight','bold');
% set(gca,'XTick',0:25:100);
% set(gca,'XTickLabel',{'0','25','50','75','100'});
% set(gca,'YTick',0:25:100);
% set(gca,'YTickLabel',{'0','25','50','75','100'});
xlabel('Number of neurons in layer 2','FontWeight','bold');
ylabel('Number of neurons in layer 1','FontWeight','bold');
zlabel('Accuracy','FontWeight','bold');
