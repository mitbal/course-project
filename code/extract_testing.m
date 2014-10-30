% Using selective search, divide test image into regions. Each region is then extracted its CNN features

% set Parameters
setParams;
addpath('SelectiveSearchCodeIJCV');
addpath('SelectiveSearchCodeIJCV/Dependencies');
addpath([params.caffePrf, 'matlab/caffe/']);

% Load all VOCImage
VOC07path = [params.dataPrf, 'VOC07/'];
annPath = [VOC07path, 'VOCdevkit/VOC2007/JPEGImages/'];
testImgs = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'], '%s');
numTest = length(testImgs);
numClasses = 20;

testF = [];
testL = zeros(numTest, numClasses);

% Load class label
for p=1:numClasses
	[~, ls] = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/', params.VOCclasses{p}, '_test.txt'], '%s %d');
	testL(:, p) = ls(:);
end

% Prepare image path
for i=1:length(testImgs)
	testImgs{i} = [annPath testImgs{i} '.jpg'];
end

% Extract the CNN features for all proposed regions
[testBoxes, testFcell] = extract_feature_all_boxes(testImgs, params);

% Convert from cell to matrix
testF = zeros(length(testFcell), length(testFcell{1}));
for i=1:length(testFcell)
	testF(i, :) = testFcell{i};
end

% Save the output to file
featureParams = [num2str(params.layerInd),'_',num2str(params.numJitter),'_',num2str(params.modelItr),'_',num2str(params.modelDataset)];
save(['../data/', params.model, '/', 'VOC07-sstest_', featureParams, '.mat'], 'testBoxes', 'testF', 'testL', '-v7.3');
