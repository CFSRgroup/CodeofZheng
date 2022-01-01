% function auc = roc_curve(deci,label_y) %%deci=wx+b, label_y, true label
%     [val,ind] = sort(deci,'descend');
%     roc_y = label_y(ind);
% %     stack_x = cumsum(roc_y == -1)/sum(roc_y == -1);
% %     stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
% %     stack_x = cumsum(roc_y == 1)/sum(roc_y == 1);
% %     stack_y = cumsum(roc_y == 3)/sum(roc_y == 3);
% 
%      stack_x = cumsum(roc_y == 1)/sum(roc_y == 1);
%      stack_y = cumsum(roc_y == 3+roc_y == 2)/sum(roc_y == 3+roc_y == 2);
%     auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));
%  
%         %Comment the above lines if using perfcurve of statistics toolbox
%         %[stack_x,stack_y,thre,auc]=perfcurve(label_y,deci,1);
%     plot(stack_x,stack_y);
%     xlabel('False Positive Rate');
%     ylabel('True Positive Rate');
%     title(['ROC curve of (AUC = ' num2str(auc) ' )']);
% end



%  yte_p       - 分类器对测试集的分类结果
%  yte  - 测试集的正确标签,这里只考虑二分类，即1，2和3
%  auc            - 返回ROC曲线的曲线下的面积
function auc = plot_roc( yte_p, yte )
%初始点为（1.0, 1.0）
%计算出ground_truth中正样本的数目pos_num和负样本的数目neg_num

% pos_num = sum(yte==1);
% neg_num = sum(yte==3);
%%%%LMW负类   MMW,HMW正类
neg_num = sum(yte==1);
pos_num = sum([(yte==2);(yte==3)]);

m=size(yte,1);
[pre,Index]=sort(yte_p);
yte=yte(Index);    %对利用模型求出的预测值yte_p由低到高进行排序;对应数据原来所在位置进行索引记录，用于重新排序yte.利用函数sort实现
x=zeros(m+1,1);
y=zeros(m+1,1);
auc=0;
x(1)=1;y(1)=1;

% for i=2:m
% TP=sum(yte(i:m)==1);FP=sum(yte(i:m)==3);
% x(i)=FP/neg_num;
% y(i)=TP/pos_num;
% auc=auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;
% end;

for i=2:m
TP=sum([(yte(i:m)==2);(yte(i:m)==3)]);FP=sum(yte(i:m)==1);
x(i)=FP/neg_num;
y(i)=TP/pos_num;
auc=auc+(y(i)+y(i-1))*(x(i-1)-x(i))/2;   %返回曲线与坐标轴间的面积ａｕｃ。我们的目的是测量数据的准确率，这个面积就是一个量度，auc越大，准确率越高。
end;

x(m+1)=0;y(m+1)=0;
auc=auc+y(m)*x(m)/2;
% plot(x,y,'-g','lineWidth',1.5);
% plot(x,y,'-b','lineWidth',1.5);
% plot(x,y,'-r','lineWidth',1.5);
plot(x,y,'-y','lineWidth',1.5);
hold on;
% plot(0:1:1,0:1:1,'-k','lineWidth',1.5);

title('(d) Case 2: session 2','FontWeight','bold');
set(gca,'XTick',0:0.2:1);
set(gca,'YTick',0:0.2:1);
xlabel('False positive rate (1-Specificity)','FontWeight','bold');
ylabel('True positive rate (Sensitivity)','FontWeight','bold');
grid on;
end