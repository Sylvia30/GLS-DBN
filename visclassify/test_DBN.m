

        rand('state',0)
        %train dbn
        dbn.sizes = [500 500 500];
        opts.numepochs =  50;
        opts.batchsize = 100;
        opts.momentum  =  0;
        opts.alpha     =   0.0001;
        dbn = dbnsetup(dbn, trainData, opts);
        dbn = dbntrain(dbn, trainData, opts);

        %unfold dbn to nn0.6
        % nn = dbnunfoldtonn(dbn, 2);
        % nn.activation_function = 'sigm';
        % 
        % %train nn
        % opts.numepochs =  100;
        % opts.batchsize = 5;
        % nn = nntrain(nn, train_x, train_y, opts);
        % [er, bad] = nntest(nn, test_x, test_y);
        % 
        % disp ('TEST ERROR IS :')
        % disp (er)
        % 
        % assert(er < 0.10, 'Too big error');



        %--------------------DBN网络参数初始化NN-----------------------------------
        input_layer_size  = size(train_x,2);  %输入数据的维数，对应可见层的数目
        hidden_layer_size1=500;
        hidden_layer_size2=500;
        hidden_layer_size3=500;
        num_labels=2;                        %所以，整个4层网络是input_layer_size-100-100-40
        % Theta1=randInitializeWeights(input_layer_size,hidden_layer_size1);
        % Theta2=randInitializeWeights(hidden_layer_size1,hidden_layer_size2);
        Theta1=[dbn.rbm{1}.c dbn.rbm{1}.W];   %训练好的DBN参数来初始化神经网络
        Theta2=[dbn.rbm{2}.c dbn.rbm{2}.W];   %
        Theta3=[dbn.rbm{3}.c dbn.rbm{3}.W];
        Theta4=randInitializeWeights(hidden_layer_size3,num_labels);%最后输出层用随机初始化
        initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:);Theta4(:)];
        lambda = 0.003;%正则化参数抑制过拟合
        %%
        %--------------------------训练神经网络-------------------------------------
        nn_params=train_nn(initial_nn_params,lambda,train_x,train_y,...
                  input_layer_size,hidden_layer_size1, hidden_layer_size2,hidden_layer_size3,num_labels);
        %%
        %------------------------对训练完后的NN进行预测和性能测试--------------------
        %将参数矩阵还原
        Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                         hidden_layer_size1, (input_layer_size + 1));

        first=1+hidden_layer_size1 * (input_layer_size + 1);
        second=hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);
        third = hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1) + hidden_layer_size3 * (hidden_layer_size1+ 1);
        Theta2 = reshape(nn_params(first:second), ...
                         hidden_layer_size2, (hidden_layer_size1 + 1));
        Theta3 = reshape(nn_params(second+1,third),...
                         hidden_layer_size3,(hidden_layer_size2 + 1));
        first=1+hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1)+ hidden_layer_size3 * (hidden_layer_size1+ 1); 
        
        
        Theta4 = reshape(nn_params(first:end), ...
                         num_labels, (hidden_layer_size3 + 1));

        %进行预测
        pred = predict(Theta1, Theta2,Theta3, Theta4,test_x);
        %计算正确率
        [dummy, expected] = max(test_y,[],2);
         bad = find(pred ~= expected);    
         er = numel(bad) / size(test_x, 1);

         %assert(er < 0.10, 'Too big error');
         s(1,i) = er;    %0.84
         
         

