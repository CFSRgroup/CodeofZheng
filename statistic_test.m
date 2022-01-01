clc;
clear;
close all;

%case1
acc_1=[0.72 0.8 0.733 0.8 0.7533 0.8667 0.78];
k_1=[0.6181 0.7254 0.6238 0.7290 0.6547 0.8239 0.7123];
acc_2=[0.6733 0.7867 0.7 0.7933 0.9133 0.7333 0.68];
k_2=[0.5175 0.7072 0.5797 0.7250 0.88 0.6434 0.5882];
% 
names=[acc_1' acc_2'];
% names=[k_1' k_2'];
% % [h,p,ci,stats]=ttest2(k_1',k_2')
% [h,p,ci,stats]=ttest(k_1',k_2')
% 
[p,table,stats]=anova1(names)

%秩和检验
% [p,h,stats]=ranksum(acc_1',acc_2');

%Wilcoxon sign-rank test
% [p,h,stats]=signrank(acc_1',acc_2')





%case2
%  yellow = [300 287 301 400 211 399 412 312 390 412];
%  red = [240 259 302 311 210 402 390 298 347 380];
%  black = [ 210 230 213 210 220 208 290 300 201 201];
%  names = [{'yellow'}; {'red'}; {'black'}];
%  [p,table,stats]=anova1([yellow' red' black'], names);
%  a=multcompare(stats);

%case3
%X = [yellow red black]';%(所有值在一个向量里)
% n_yellow=repmat({'yellow'},10,1);
% n_red=repmat({'red'},10,1);
% n_black=repmat({'black'},10,1);
%  group= [n_yellow' n_red' n_black']';%(group里则是相对于的X中值的类型)
% [p,table,stats]=anova1(X,group);

%case4
% % p = anova1(X,GROUP,DISPLAYOPT); %其中DISPLAYOPT是用来控制图表的展现的，可以设置成off,也可设成on. 
% X = [yellow red black]';%(所有值在一个向量里)
% n_yellow=repmat({'yellow'},10,1);
% n_red=repmat({'red'},10,1);
% n_black=repmat({'black'},10,1);
% group= [n_yellow' n_red' n_black']';%(group里则是相对于的X中值的类型)
% p = anova1(X,group,'on'); 

%case5
% （2）非均衡数据  
% 处理非均衡数据的用法为： 
% p=anova1(x,group) 

% x=[1620 1580 1460 1500
%    1670 1600 1540 1550 
%    1700 1640 1620 1610
%    1750 1720 1680 1800]; 
% 
% x=[x(1:4),x(16),x(5:8),x(9:11),x(12:15)]; 
% 
% g=[ones(1,5),2*ones(1,4),3*ones(1,3),4*ones(1,4)]; 
% p=anova1(x,g); 

%mean+std
% x=[0.5796 0.537 0.5741 0.4611 0.65 0.4963 0.5185 0.5481 0.5519 0.4296 0.5056 0.5593 0.6593 0.5907 0.5111 0.5370];
% mean_x=sum(x)/16;
% y=std(x);


%matlab 条形图绘制 以及 添加误差棒 改变条形图形状
% a_live = [0.9186, 0.9460, 0.9552, 0.9533];
% a_tid = [0.6090, 0.6663, 0.7170, 0.7165];
% a = [a_live; a_tid];
% bar(a, 'grouped')
% set(gca,'YLim', [0.5,1], 'XTickLabel',{'LIVE', 'TID2013'}, 'FontSize', 15);
% ylabel('SRC');
% set(gca, 'Ytick', [0.5:0.05:1], 'ygrid','on','GridLineStyle','-');
% legend('25','50','100','200', 'Location', 'EastOutside');
% legend('boxoff');
% 
% e = [0.0198, 0.0124, 0.0096, 0.0112; 0.0875, 0.0990, 0.1034, 0.0939];
% hold on
% numgroups = size(a, 1); 
% numbars = size(a, 2); 
% groupwidth = min(0.8, numbars/(numbars+1.5));
% for i = 1:numbars
%       % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
%       x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
%       errorbar(x, a(:,i), e(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
% end



