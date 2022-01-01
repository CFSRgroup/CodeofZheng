%function：pearson correlation coefficient
%function：spearman correlation coefficient
%function：kendall correlation coefficient
%author：郑展鹏
%2019/04/03

% % 单通道特征8位被试session1
% clc;
% clear;
% close all;
% warning off;
% 
% for pp=1:8
% eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_1'])
% 
% Y=y(:,1);
% 
% for j=1:11
%     X=x(:,[1+(j-1)*4:4+(j-1)*4 61+(j-1)*10:70+(j-1)*10]);
% %     coeff(j+(pp-1)*11,:)=corr(X,Y)'; %pearson相关系数    
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
%     coeff(j+(pp-1)*11,:)=corr(X,Y,'type','kendall')'; %kendall相关系数  
% end
% end
% 
% for i=1:11
% coeff_mean(i,:)=(coeff(1+(i-1),:)+coeff(12+(i-1),:)+coeff(23+(i-1),:)+coeff(34+(i-1),:)+coeff(45+(i-1),:)+coeff(56+(i-1),:)+coeff(67+(i-1),:)+coeff(78+(i-1),:))/8;
% end
% 
% 
% %单通道特征8位被试session2
% clc;
% clear;
% close all;
% warning off;
% 
% for pp=1:8
% eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_2'])
% 
% Y=y(:,1);
% 
% for j=1:11
%     X=x(:,[1+(j-1)*4:4+(j-1)*4 61+(j-1)*10:70+(j-1)*10]);
% %     coeff(j+(pp-1)*11,:)=corr(X,Y)'; %pearson相关系数    
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
%     coeff(j+(pp-1)*11,:)=corr(X,Y,'type','kendall')'; %kendall相关系数  
% end
% end
% 
% for i=1:11
% coeff_mean(i,:)=(coeff(1+(i-1),:)+coeff(12+(i-1),:)+coeff(23+(i-1),:)+coeff(34+(i-1),:)+coeff(45+(i-1),:)+coeff(56+(i-1),:)+coeff(67+(i-1),:)+coeff(78+(i-1),:))/8;
% end

%单通道特征6位被试session1
% clc;
% clear;
% close all;
% warning off;
% 
% for pp=1:6
% eval(['load F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(pp) '_1'])
% 
% Y=y(:,1);
% 
% for j=1:11
%     X=x(:,[1+(j-1)*4:4+(j-1)*4 61+(j-1)*10:70+(j-1)*10]);
%     coeff(j+(pp-1)*11,:)=corr(X,Y)'; 
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','kendall')'; %kendall相关系数 
% end
% end
% 
% for i=1:11
% coeff_mean(i,:)=(coeff(1+(i-1),:)+coeff(12+(i-1),:)+coeff(23+(i-1),:)+coeff(34+(i-1),:)+coeff(45+(i-1),:)+coeff(56+(i-1),:))/6;
% end

%单通道特征6位被试session2
% clc;
% clear;
% close all;
% warning off;
% 
% for pp=1:6
% eval(['load F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(pp) '_2'])
% 
% Y=y(:,1);
% 
% for j=1:11
%     X=x(:,[1+(j-1)*4:4+(j-1)*4 61+(j-1)*10:70+(j-1)*10]);
%     coeff(j+(pp-1)*11,:)=corr(X,Y)';    
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','kendall')'; %kendall相关系数 
% end
% end
% 
% for i=1:11
% coeff_mean(i,:)=(coeff(1+(i-1),:)+coeff(12+(i-1),:)+coeff(23+(i-1),:)+coeff(34+(i-1),:)+coeff(45+(i-1),:)+coeff(56+(i-1),:))/6;
% end


%双通道特征8位被试session1
% clc;
% clear;
% close all;
% warning off;
% 
% for pp=1:8
% eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_1'])
% 
% Y=y(:,1);
% 
% for j=1:4
%     X=x(:,45+(j-1)*4:48+(j-1)*4);
%     coeff(j+(pp-1)*4,:)=corr(X,Y)';   
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','kendall')'; %kendall相关系数 
% end
% end
% 
% for i=1:4
% coeff_mean(i,:)=(coeff(1+(i-1),:)+coeff(5+(i-1),:)+coeff(9+(i-1),:)+coeff(13+(i-1),:)+coeff(17+(i-1),:)+coeff(21+(i-1),:)+coeff(25+(i-1),:)+coeff(29+(i-1),:))/8;
% end


%双通道特征8位被试session2
% clc;
% clear;
% close all;
% warning off;
% 
% for pp=1:8
% eval(['load F:\matlab\trial_procedure\study_1\features\ex1\s' num2str(pp) '_2'])
% 
% Y=y(:,1);
% 
% for j=1:4
%     X=x(:,45+(j-1)*4:48+(j-1)*4);
%     coeff(j+(pp-1)*4,:)=corr(X,Y)';  
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','kendall')'; %kendall相关系数 
% end
% end
% 
% for i=1:4
% coeff_mean(i,:)=(coeff(1+(i-1),:)+coeff(5+(i-1),:)+coeff(9+(i-1),:)+coeff(13+(i-1),:)+coeff(17+(i-1),:)+coeff(21+(i-1),:)+coeff(25+(i-1),:)+coeff(29+(i-1),:))/8;
% end


%双通道特征6位被试session1
% clc;
% clear;
% close all;
% warning off;
% 
% for pp=1:6
% eval(['load F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(pp) '_1'])
% 
% Y=y(:,1);
% 
% for j=1:4
%     X=x(:,45+(j-1)*4:48+(j-1)*4);
%     coeff(j+(pp-1)*4,:)=corr(X,Y)';  
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','kendall')'; %kendall相关系数 
% end
% end
% 
% for i=1:4
% coeff_mean(i,:)=(coeff(1+(i-1),:)+coeff(5+(i-1),:)+coeff(9+(i-1),:)+coeff(13+(i-1),:)+coeff(17+(i-1),:)+coeff(21+(i-1),:))/6;
% end


%双通道特征6位被试session2
% clc;
% clear;
% close all;
% warning off;
% 
% for pp=1:6
% eval(['load F:\matlab\trial_procedure\study_1\features\ex2\s' num2str(pp) '_2'])
% 
% Y=y(:,1);
% 
% for j=1:4
%     X=x(:,45+(j-1)*4:48+(j-1)*4);
%     coeff(j+(pp-1)*4,:)=corr(X,Y)'; 
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','Spearman')'; %spearman相关系数   
%     %coeff(j+(pp-1)*11,:)=corr(X,Y,'type','kendall')'; %kendall相关系数 
% end
% end
% 
% for i=1:4
% coeff_mean(i,:)=(coeff(1+(i-1),:)+coeff(5+(i-1),:)+coeff(9+(i-1),:)+coeff(13+(i-1),:)+coeff(17+(i-1),:)+coeff(21+(i-1),:))/6;
% end