%%
clc;clear;close all;
%% load
load DRdata2.mat
%%
DRdata = DRdata2';
DRdata = DRdata(:,1:287);         % AD,MCI : 350*(147+140)
data = DRdata';
label = zeros(287,2);               % 287 * 2
label(1:147,2) = ones(147,1);       % �ڶ���ǰ140����Ϊ1
label(148:287,1) = ones(140,1);     % �ڶ��к�430����Ϊ1

randIndex = randperm(287);
data_new = data(randIndex,:);
label_new = label(randIndex,:);
data_new = mapminmax(data_new',0,1)';
label1 = label_new(:,1)';        % ��ȡ��һ�У�NCΪ1

[M,N]=size(data_new);
indices=crossvalind('Kfold',data_new(1:M,N),10);%�������N����������K=10�۽�����֤������ɵ�������
                                % Indices��1~K������������K��ƽ������ӽ�ƽ�������Ӽ�
result = zeros(2,10);
%%
pro = {};
% data1 = {};
% k=2;
for k=1:10%10
        %9��train.1��test
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
       save data2 trainData trainLabels testData testLabels
       [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,10,'quadratic');
        result(1,k) = errorBeforeBP;
        result(2,k) = errorAfterBP;
        pro{k} = y; 
%          delete data2.mat
end
%%
% 
% %%
acc = 1- mean(result(2,:));
 


