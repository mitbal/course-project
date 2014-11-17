% Weakly supervised pre-experiment

% Run selective search to get segment
setParams;
addpath('/home/iqbal/caffe/matlab/caffe/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/VOCcode/');

% Load all training images
VOC07path = '/home/iqbal/data/VOC07/';
imdirpath = [VOC07path, 'VOCdevkit/VOC2007/JPEGImages/'];
train_imgs = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'], '%s');
annopath = [VOC07path, 'VOCdevkit/VOC2007/Annotations/'];
num_train = length(train_imgs);

% Run selective search to get segment annotation
count_boxes = zeros(num_train, 1);
alL_boxes = [];
for ii=1:num_train
	impath = [imdirpath train_imgs{ii} '.jpg'];
	im = imread(impath);
	boxes = selective_search(im);
	count_boxes = size(boxes, 1);
	all_boxes = [all_boxes; boxes];
end

% Load all training data that has been produced on previous experiment
% also load the model to get decision value
load(['../data/caffe/VOC07-sstrain_7_16_0_imagenet.mat']);
load(['../models/caffe/VOC07-ss_7_16_0_imagenet.mat']);

% Normalization
disp(['Normalization...']);
tic
for i=1:size(trainF, 1)
    trainF(i, :) = sign(trainF(i, :)) .* abs(trainF(i, :)).^2;
end
toc
train_sums = sum(abs(trainF));
tic
for i=1:size(trainF, 1)
    trainF(i, :) = trainF(i, :) ./ train_sums;
end
toc
tic
for i=1:size(trainF, 1)
    trainF(i, :) = trainF(i, :) ./ sqrt(sum(trainF(i, :).^2));
end
toc

num_class = 1;
num_data = size(trainF, 2);
for ii=1:num_class
	model = models{ii, 6};
	dec_value = model.w * trainF(:, :)';

	% Sort based on decision value
	abc = [(1:num_data)' dec_value];
	abc = sortrows(abc, 2);

	% Get the top result
	for jj=1:10
		ind = abc(jj, 1);
		box = all_boxes(ind, :);
		sum_box = cumsum(count_boxes);
		for kk=1:num_train
			if ind > sum_box(kk)
				imdex = kk;
				break
			end
		end
		impath = [imdirpath train_imgs{imdex} '.jpg'];
		im = imread(impath);
		patch = im(box(1):box(3), box(2):box(4));
		imshow(patch);
	end
end
