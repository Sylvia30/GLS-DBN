%%
%交叉验证DBN

indices = crossvalind('Kfold', label_new(:,1), 5); %K折交叉验证

% 
% [Train, Test] = crossvalind('HoldOut', N, P) % 将原始数据随机分为两组,一组做为训练集,一组做为验证集
% 
% [Train, Test] = crossvalind('LeaveMOut', N, M) %留M法交叉验证，默认M为1，留一法交叉验证

acc = zeros(1,5);

for i = 1:5   %实验记进行5次(交叉验证折数)，求5次的平均值作为实验结果，
    test = (indices == i); 
    train = ~test;  %产生测试集合训练集索引
%     train_x=data_new(train,:);
%     train_y=label_new(train,:);
    train_x=data_new;
    train_y=label_new;
    
    test_x=data(test,:);
    test_y=label_new(test,:);
    
    %---------------设置DBN网络的各种参数---------------------------------------
    rand('state',0)
    %train dbn
    dbn.sizes = [100 100 ];
    opts.numepochs =   30;
    opts.batchsize = 1;
    opts.momentum  =  0.1;
    opts.alpha     =   0.001;
    dbn = dbnsetup(dbn, train_x, opts);
    dbn = dbntrain(dbn, train_x, opts);
    
    %%
    %--------------------DBN网络参数初始化NN-----------------------------------
    input_layer_size  = size(train_x,2);  %输入数据的维数，对应可见层的数目
    hidden_layer_size1=100;
    hidden_layer_size2=100;
    num_labels=2;                        %所以，整个4层网络是input_layer_size-100-100-40
    % Theta1=randInitializeWeights(input_layer_size,hidden_layer_size1);
    % Theta2=randInitializeWeights(hidden_layer_size1,hidden_layer_size2);
    Theta1=[dbn.rbm{1}.c dbn.rbm{1}.W];   %训练好的DBN参数来初始化神经网络
    Theta2=[dbn.rbm{2}.c dbn.rbm{2}.W];   %
    Theta3=randInitializeWeights(hidden_layer_size2,num_labels);%最后输出层用随机初始化
    initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:)];
    lambda = 0.003;%正则化参数抑制过拟合
    
    %%
    %--------------------------训练神经网络-------------------------------------
    nn_params=train_nn(initial_nn_params,lambda,train_x,train_y,...
              input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);
    %%
    %------------------------对训练完后的NN进行预测和性能测试--------------------
    %将参数矩阵还原
    Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                     hidden_layer_size1, (input_layer_size + 1));

    first=1+hidden_layer_size1 * (input_layer_size + 1);
    second=hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);
    Theta2 = reshape(nn_params(first:second), ...
                     hidden_layer_size2, (hidden_layer_size1 + 1));

    first=1+hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);     
    Theta3 = reshape(nn_params(first:end), ...
                     num_labels, (hidden_layer_size2 + 1));

    %进行预测
    pred = predict(Theta1, Theta2,Theta3, test_x);
    %计算正确率
    [dummy, expected] = max(test_y,[],2);
    bad = find(pred ~= expected);    
    er = numel(bad) / size(test_x, 1);
    
    acc(1,i) = 1-er;
     
end









