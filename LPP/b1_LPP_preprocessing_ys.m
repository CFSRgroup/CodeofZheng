clc;
clear;
close all;

for pp=1:8
 rr=120
%%
%载入数据
eval(['load D:\exp\features\ex1\s' num2str(pp) '_1'])
x1=x;
y1=y;
eval(['load D:\exp\features\ex1\s' num2str(pp) '_2'])
x2=x;
y2=y;
x=[x1;x2];
y=[y1;y2];
%%
%进行LPP
fea =x;
options = [];
options.Metric = 'Euclidean';
% options.NeighborMode = 'Supervised';
% options.gnd = y;
options.ReducedDim=rr;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x=fea*eigvector;

eval(['save D:\exp\LPP_preprocessed3\s' num2str(rr) num2str(pp) ' x y'])
% eval(['save D:\exp\c3_construct\neww\s' num2str(pp) ' eigvector'])
end
