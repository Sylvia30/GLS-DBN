%%
%������֤DBN

indices = crossvalind('Kfold', label_new(:,1), 5); %K�۽�����֤

% 
% [Train, Test] = crossvalind('HoldOut', N, P) % ��ԭʼ���������Ϊ����,һ����Ϊѵ����,һ����Ϊ��֤��
% 
% [Train, Test] = crossvalind('LeaveMOut', N, M) %��M��������֤��Ĭ��MΪ1����һ��������֤

acc = zeros(1,5);

for i = 1:5   %ʵ��ǽ���5��(������֤����)����5�ε�ƽ��ֵ��Ϊʵ������
    test = (indices == i); 
    train = ~test;  %�������Լ���ѵ��������
%     train_x=data_new(train,:);
%     train_y=label_new(train,:);
    train_x=data_new;
    train_y=label_new;
    
    test_x=data(test,:);
    test_y=label_new(test,:);
    
    %---------------����DBN����ĸ��ֲ���---------------------------------------
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
    %--------------------DBN���������ʼ��NN-----------------------------------
    input_layer_size  = size(train_x,2);  %�������ݵ�ά������Ӧ�ɼ������Ŀ
    hidden_layer_size1=100;
    hidden_layer_size2=100;
    num_labels=2;                        %���ԣ�����4��������input_layer_size-100-100-40
    % Theta1=randInitializeWeights(input_layer_size,hidden_layer_size1);
    % Theta2=randInitializeWeights(hidden_layer_size1,hidden_layer_size2);
    Theta1=[dbn.rbm{1}.c dbn.rbm{1}.W];   %ѵ���õ�DBN��������ʼ��������
    Theta2=[dbn.rbm{2}.c dbn.rbm{2}.W];   %
    Theta3=randInitializeWeights(hidden_layer_size2,num_labels);%���������������ʼ��
    initial_nn_params = [Theta1(:) ; Theta2(:);Theta3(:)];
    lambda = 0.003;%���򻯲������ƹ����
    
    %%
    %--------------------------ѵ��������-------------------------------------
    nn_params=train_nn(initial_nn_params,lambda,train_x,train_y,...
              input_layer_size,hidden_layer_size1, hidden_layer_size2,num_labels);
    %%
    %------------------------��ѵ������NN����Ԥ������ܲ���--------------------
    %����������ԭ
    Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                     hidden_layer_size1, (input_layer_size + 1));

    first=1+hidden_layer_size1 * (input_layer_size + 1);
    second=hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);
    Theta2 = reshape(nn_params(first:second), ...
                     hidden_layer_size2, (hidden_layer_size1 + 1));

    first=1+hidden_layer_size1 * (input_layer_size + 1)+hidden_layer_size2 * (hidden_layer_size1+ 1);     
    Theta3 = reshape(nn_params(first:end), ...
                     num_labels, (hidden_layer_size2 + 1));

    %����Ԥ��
    pred = predict(Theta1, Theta2,Theta3, test_x);
    %������ȷ��
    [dummy, expected] = max(test_y,[],2);
    bad = find(pred ~= expected);    
    er = numel(bad) / size(test_x, 1);
    
    acc(1,i) = 1-er;
     
end









