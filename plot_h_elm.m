clc;
clear;
close all;
warning off;

%(a)case1_session1
for pp=2:11
eval(['load F:\matlab\trial_procedure\study_1\number_of_abstraction_layers_H-ELM\case1_session1_layer' num2str(pp)])
Y1(pp-1,:)=Acc_mean(:,1)';
end

%(b)case1_session2
for pp=2:11
eval(['load F:\matlab\trial_procedure\study_1\number_of_abstraction_layers_H-ELM\case1_session2_layer' num2str(pp)])
Y2(pp-1,:)=Acc_mean(:,1)';
end

%(c)case2_session1
for pp=2:11
eval(['load F:\matlab\trial_procedure\study_1\number_of_abstraction_layers_H-ELM\case2_session1_layer' num2str(pp)])
Y3(pp-1,:)=Acc_mean_1(:,1)';
end

%(d)case2_session2
for pp=2:11
eval(['load F:\matlab\trial_procedure\study_1\number_of_abstraction_layers_H-ELM\case2_session2_layer' num2str(pp)])
Y4(pp-1,:)=Acc_mean_1(:,1)';
end

y1=(Y1+Y2)./2;
y2=(Y3+Y4)./2;
[a,b]=find(y1==max(max(y1)));
[c,d]=find(y2==max(max(y2)));

% subplot(4,1,1);
% bar(Y1,'DisplayName','Y1'); 
% set(gca,'XTickLabel',{'2','3','4','5','6','7','8','9','10','11'});
% % set(gca,'YTick',0:0.05:0.5);
% xlabel('Number of abstraction layers','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% title('(a) Case 1: Session 1','FontWeight','bold');
% grid on;
% legend('NOHN=10','NOHN=20','NOHN=30','NOHN=40','NOHN=50','NOHN=60','NOHN=70','NOHN=80','NOHN=90','NOHN=100');
% 
% subplot(4,1,2);
% bar(Y2,'DisplayName','Y2'); 
% set(gca,'XTickLabel',{'2','3','4','5','6','7','8','9','10','11'});
% % set(gca,'YTick',0:0.05:0.5);
% xlabel('Number of abstraction layers','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% title('(b) Case 1: Session 2','FontWeight','bold');
% grid on;
% 
% subplot(4,1,3);
% bar(Y3,'DisplayName','Y3'); 
% set(gca,'XTickLabel',{'2','3','4','5','6','7','8','9','10','11'});
% % set(gca,'YTick',0:0.05:0.5);
% xlabel('Number of abstraction layers','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% title('(c) Case 2: Session 1','FontWeight','bold');
% grid on;
% 
% subplot(4,1,4);
% bar(Y4,'DisplayName','Y4'); 
% set(gca,'XTickLabel',{'2','3','4','5','6','7','8','9','10','11'});
% % set(gca,'YTick',0:0.05:0.5);
% xlabel('Number of abstraction layers','FontWeight','bold');
% ylabel('Accuracy','FontWeight','bold');
% title('(d) Case 2: Session 2','FontWeight','bold');
% grid on;