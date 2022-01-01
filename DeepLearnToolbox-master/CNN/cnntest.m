% function [er, bad] = cnntest(net, x, y)
function [h, a] = cnntest(net, x, y)
    %  feedforward
    net = cnnff(net, x);
%     [~, h] = max(net.o);
%     [~, a] = max(y);
    h = round(net.o);
    a = y;
%     bad = find(h ~= a);
% 
%     er = numel(bad) / size(y, 2);
end
