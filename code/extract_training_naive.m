% Main script to run selective search for classification

% set Parameters
setParams;
addpath('SelectiveSearchCodeIJCV');
addpath('SelectiveSearchCodeIJCV/Dependencies');
addpath([params.caffePrf, 'matlab/caffe/']);
addpath('/home/iqbal/liblinear-1.93/matlab');

% Load all VOCImage
VOC07path = [params.dataPrf, 'VOC07/'];
annPath = [VOC07path, 'VOCdevkit/VOC2007/JPEGImages/'];
trainImgs = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'], '%s');
numTrain = length(trainImgs);
numClasses = 20;

trainL = zeros(numTrain, numClasses);

% Load class label
for p=1:numClasses
	[~, ls] = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/', params.VOCclasses{p}, '_trainval.txt'], '%s %d');
	trainL(:, p) = ls(:);
end

% Prepare image path
for i=1:length(trainImgs)
	trainImgs{i} = [annPath trainImgs{i} '.jpg'];
end

% Extract the CNN features for all proposed regions
[trainBoxes, trainFcell] = extract_feature_all_boxes(trainImgs, params);

% Convert from cell to matrix
trainF = zeros(length(trainFcell), length(trainFcell{1}));

for i=1:length(trainFcell)
	trainF(i, :) = trainFcell{i};
end

% Save the output to file
featureParams = [num2str(params.layerInd),'_',num2str(params.numJitter),'_',num2str(params.modelItr),'_',num2str(params.modelDataset)];
save(['../data/', params.model,'/', 'VOC07-sstrain_', featureParams,'.mat'], 'trainBoxes', 'trainF', 'trainL', '-v7.3');
