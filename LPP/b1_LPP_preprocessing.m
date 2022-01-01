clc;
clear;
close all;

for pp=1:8
%%
%载入数据
eval(['load sample' num2str(pp) '_1'])
%%
%进行LPP
fea =x;
options = [];
options.Metric = 'Euclidean';
options.NeighborMode = 'Supervised';
options.gnd = y;
options.ReducedDim=27;
W = constructW(fea,options);      
options.PCARatio = 1;
[eigvector, eigvalue] = LPP(W, options, fea);
x=fea*eigvector;

eval(['save S' num2str(pp) ' x y'])
end