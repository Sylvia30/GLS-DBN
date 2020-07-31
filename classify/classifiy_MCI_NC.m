%%
clc;clear;close all;
%% load
load DRdata.mat
%%
DRdata = DRdata(:,148:717);         % MCI NC: 350*570
data = DRdata';
label = zeros(570,2);               % 570 * 2
label(1:140,2) = ones(140,1);       % 第二列前140个令为1
label(141:570,1) = ones(430,1);     % 第二列后430个令为1

randIndex = randperm(570);
data_new = data(randIndex,:);
label_new = label(randIndex,:);
data_new = mapminmax(data_new',0,1)';
label1 = label_new(:,1)';        % 提取第一列：NC为1
%indices=crossvalind('Kfold',358,5);

[M,N]=size(data_new);
indices=crossvalind('Kfold',data_new(1:M,N),10);%返回针对N个样本进行K=10折交叉验证随机生成的索引，
                                % Indices是1~K的整数，代表K个平均（或接近平均）的子集
result = zeros(2,10);
%%
pro = {};
% data1 = {};
trainLabels=[];
testLabels=[];
trainData=[];
testData=[];

% k=4;
for k=1:10
        test = (indices == k);   
        train = ~test;
        trainData=data_new(train,:);
        trainLabels=zeros(sum(train),2);
        b=find(train==1);
        for i=1:length(b)
            c = b(i);
            trainLabels(i,:)=label_new(c,:);
        end
        
%         trainLabels=label_new(:,train)';
        testData=data_new(test,:);
        testLabels=zeros(sum(test),2);
        d=find(test==1);
        for j=1:length(d)
            e = d(j);
            testLabels(j,:)=label_new(e,:);
        end
        
%        save data1 trainData trainLabels testData testLabels
       [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,10,'quadratic');%%%%%%%%%%%%%bug
        result(1,k) = errorBeforeBP;
        result(2,k) = errorAfterBP;
        pro{k} = y; 
%         delete data1.mat
end
%%

%%
acc = 1- mean(result(2,:));
 


