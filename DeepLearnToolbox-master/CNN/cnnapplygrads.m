function net = cnnapplygrads(net, opts)   %卷积层及全连接层权重，偏置均等于原来值减去学习率乘以相应的梯度
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                end
                net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW; %单层感知机更新后的权重参数
    net.ffb = net.ffb - opts.alpha * net.dffb; %单层感知机更新后的偏置参数
end
