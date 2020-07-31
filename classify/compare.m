%%
%对比试验:分别讨论隐层节点数目，参数设置 ， 迭代次数 对分类结果的影响

%%
%加载数据 每次均做十折交叉验证

load DRdata.mat

data = DRdata';
label = zeros(358,2);
label(1:103,2) = ones(103,1);
label(104:358,1) = ones(255,1);
randIndex = randperm(358);
data_new = data(randIndex,:);
label_new = label(randIndex,:);
data_new = mapminmax(data_new',0,1)';
label_new = label_new(:,1)';
%indices=crossvalind('Kfold',358,5);

[M,N]=size(data_new);
indices=crossvalind('Kfold',data_new(1:M,N),10);



%%
%对比试验：讨论迭代次数对实验结果的影响
epoch = 50;
result = zeros(6,epoch);
for i = 1:epoch
    %交叉验证
    
    fprintf('epoch :------------------');
    num2str(i)
    for k=1:10
        test = (indices == k);   
        train = ~test;
        trainData=data_new(train,:);
        trainLabels=label_new(:,train)';
        testData=data_new(test,:);
        testLabels=label_new(:,test)';
        %quadratic
        [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,i,'quadratic');
        tempresult(1,k) = errorAfterBP;
        %rateDistortion
        [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,i,'rateDistortion');
        tempresult(2,k) = errorAfterBP;
        %normal
        [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,i,'normal');
        tempresult(3,k) = errorAfterBP;
        %Laplace
        [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,i,'Laplace');
        tempresult(4,k) = errorAfterBP;
        %cauchy
        [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,i,'cauchy');
        tempresult(5,k) = errorAfterBP;
        %
        [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,i,'lasso');
        tempresult(6,k) = errorAfterBP;
    end
    result(:,i) = mean(tempresult,2);
    

    
end



%%
%对比试验：讨论隐层节点的数目对实验结果的影响





