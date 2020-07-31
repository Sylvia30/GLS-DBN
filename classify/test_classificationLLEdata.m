
function [errorBeforeBP,errorAfterBP,y] = test_classificationLLEdata(trainData,testData,trainLabels,testLabels,epoch,spraseMethod)

%Test classification in DBN in MNIST data set
%clc;
%clear all;

res={};
more off;
addpath(genpath('DeepLearnToolboxGPU'));
addpath('DeeBNet');



%data = MNIST.prepareMNIST('');%using MNIST dataset completely.
data = MNIST.prepareMNIST_Small('+MNIST\');%uncomment this line to use a small part of MNIST dataset.
data.normalize('minmax');
data.shuffle();
% 
% load data1.mat;
% load data2.mat;
data.trainData = trainData;
data.testData = testData;
data.trainLabels = trainLabels(:,1);
data.testLabels = testLabels(:,1);
data.validationData=data.testData;
data.validationLabels=data.testLabels;

dbn=DBN('classifier');
% RBM1   libs/DeeBNetV3.1/DeeBNet/tools/RbmParameters.m
rbmParams=RbmParameters(500,ValueType.binary);  % 500Òþ²Ø²ãµ¥ÔªÊý
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='reconstruction';
rbmParams.sparsityMethod = spraseMethod;
rbmParams.maxEpoch=epoch;
dbn.addRBM(rbmParams);
% RBM2
rbmParams=RbmParameters(500,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.performanceMethod='reconstruction';
rbmParams.sparsityMethod = spraseMethod;
rbmParams.maxEpoch=epoch;
dbn.addRBM(rbmParams);
% RBM3
rbmParams=RbmParameters(500,ValueType.binary);
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.PCD;
rbmParams.maxEpoch=epoch;
rbmParams.rbmType=RbmType.discriminative;
rbmParams.performanceMethod='classification';
rbmParams.sparsityMethod = spraseMethod;
dbn.addRBM(rbmParams);

%train  libs/DeeBNetV3.1/DeeBNet/tools/DBN.m
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

end
