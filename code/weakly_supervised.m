% Weakly supervised pre-experiment

% Run selective search to get segment
setParams;
addpath('/home/iqbal/caffe/matlab/caffe/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/VOCcode/');
addpath('SelectiveSearchCodeIJCV/');
addpath('SelectiveSearchCodeIJCV/Dependencies/');

% Load all training images
VOC07path = '/home/iqbal/data/VOC07/';
imdirpath = [VOC07path, 'VOCdevkit/VOC2007/JPEGImages/'];
train_imgs = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'], '%s');
annopath = [VOC07path, 'VOCdevkit/VOC2007/Annotations/'];
num_train = length(train_imgs);

%% Run selective search to get segment annotation
count_boxes = zeros(num_train, 1);
all_boxes = [];
tic
disp(['Selective search']);
for ii=1:num_train
    tic
	impath = [imdirpath train_imgs{ii} '.jpg'];
	im = imread(impath);
	boxes = selective_search(im);
	count_boxes(ii) = size(boxes, 1);
	all_boxes = [all_boxes; boxes];
    disp(['Image: ' num2str(ii) ' boxes: ' num2str(count_boxes(ii))]);
    toc
end
disp(['Total regions: ' num2str(sum(count_boxes))]);
toc

save('../models/caffe/box-VOC07-sstrain_7_16_0_imagenet.mat', 'all_boxes', 'count_boxes');

%% Load all training data that has been produced on previous experiment
% also load the model to get decision value
tic
disp(['Load training data and model']);
load(['../data/caffe/VOC07-sstrain_7_16_0_imagenet.mat']);
load(['../models/caffe/VOC07-ss_7_16_0_imagenet.mat']);
toc

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

%% Create image mosaic
num_class = 1;
num_data = size(trainF, 1);
for ii=1:20
    tic
    disp(['Processing class: ' num2str(ii)]);
	model = models{ii, 6};
	dec_value = model.w * trainF(:, :)';

	% Sort based on decision value
	abc = [(1:num_data)' dec_value'];
	abc = sortrows(abc, 2);

	% Get the top negative
    results = uint8(zeros(2000, 2000, 3));
	for y=0:19
        for x=0:9
            jj = y*10+x+1;
            ind = abc(jj, 1);
            box = all_boxes(ind, :);
            sum_box = cumsum(count_boxes);
            for kk=1:num_train
                if sum_box(kk) > ind
                    imdex = kk;
                    break
                end
            end
            impath = [imdirpath train_imgs{imdex} '.jpg'];
            im = imread(impath);
            patch = im(box(1):box(3), box(2):box(4), :);
            patch = imresize(patch, [200 200]);
            %figure
            %imshow(patch);
            results(y*200+1:(y+1)*200, x*200+1:(x+1)*200, :) = patch(:, :, :);
        end
    end
    %figure
    %imshow(results);
    imwrite(results, ['../plot/class_' num2str(ii) '_neg.jpg']);
    
    % Get the top positive, reverse the list
    abc = abc(end:-1:1, :);
    results = uint8(zeros(2000, 2000, 3));
	for y=0:19
        for x=0:9
            jj = y*10+x+1;
            ind = abc(jj, 1);
            box = all_boxes(ind, :);
            sum_box = cumsum(count_boxes);
            for kk=1:num_train
                if sum_box(kk) > ind
                    imdex = kk;
                    break
                end
            end
            impath = [imdirpath train_imgs{imdex} '.jpg'];
            im = imread(impath);
            patch = im(box(1):box(3), box(2):box(4), :);
            patch = imresize(patch, [200 200]);
            %figure
            %imshow(patch);
            results(y*200+1:(y+1)*200, x*200+1:(x+1)*200, :) = patch(:, :, :);
        end
    end
    %figure
    %imshow(results);
    imwrite(results, ['../plot/class_' num2str(ii) '_pos.jpg']);
    toc
end
