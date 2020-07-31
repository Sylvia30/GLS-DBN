%œ° ËDBN 
clc;
clear all;

res={};
more off;
addpath(genpath('DeepLearnToolboxGPU'));
addpath('DeeBNet');

data = MNIST.prepareMNIST_Small('+MNIST\');%uncomment this line to use a small part of MNIST dataset.
data.normalize('minmax');
data.shuffle();


load data10_9.mat;

data.trainData = trainData;
data.testData = testData;
data.trainLabels = trainLabels(:,1);
data.testLabels = testLabels(:,1);
data.validationData=data.testData;
data.validationLabels=data.testLabels;

dbn=DBN('classifier');
% RBM1
rbmParams=RbmParameters(500,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='reconstruction';
%rbmParams.sparsity=0
% rbmParams.sparsityMethod = 'lasso';
rbmParams.maxEpoch=50;
dbn.addRBM(rbmParams);
% RBM2
rbmParams=RbmParameters(500,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='reconstruction';
rbmParams.sparsity=0
% rbmParams.sparsityMethod = 'lasso';
%rbmParams.maxEpoch=50;
dbn.addRBM(rbmParams);
% RBM3
rbmParams=RbmParameters(500,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.maxEpoch=50;
rbmParams.rbmType=RbmType.discriminative;
rbmParams.performanceMethod='classification';
%rbmParams.sparsity=0
% rbmParams.sparsityMethod = 'lasso';
dbn.addRBM(rbmParams);
%train
ticID=tic;
dbn.train(data);
toc(ticID)
%test train
[y,classNumber]=dbn.getOutput(data.testData,'bySampling');
errorBeforeBP=sum(classNumber~=data.testLabels)/length(classNumber)
%BP
ticID=tic;
dbn.backpropagation(data);
toc(ticID);
%test after BP
[y,classNumber]=dbn.getOutput(data.testData);
errorAfterBP=sum(classNumber~=data.testLabels)/length(classNumber)