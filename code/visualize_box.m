%% Visualize top k bounding box in positive image

setParams;
addpath('/home/iqbal/caffe/matlab/caffe/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/VOCcode/');
addpath('SelectiveSearchCodeIJCV/');
addpath('SelectiveSearchCodeIJCV/Dependencies/');

% Load all training images and necessary models
VOC07path = '/home/iqbal/data/VOC07/';
imdirpath = [VOC07path, 'VOCdevkit/VOC2007/JPEGImages/'];
train_imgs = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'], '%s');
annopath = [VOC07path, 'VOCdevkit/VOC2007/Annotations/'];
num_train = length(train_imgs);
trainL = zeros(num_train, 20);
for p = 1 : numClasses
    [~, ls] = textread([VOC07Path,'VOCdevkit/VOC2007/ImageSets/Main/' params.VOCclasses{p} '_trainval.txt'], '%s %d');
    trainL(:, p) = ls(:);
end

load('../models/caffe/VOC07-ss.mat');

k = 3;
n = 10;
num_class = 20;

for ii=1:num_class
	index = trainL(:, ii) > 1;
	pos_images = train_imgs{index};
	for jj=1:n
		impath = [imdirpath pos_images{jj} '.jpg'];
		im = imread(impath);

		boxes = selective_search(im);
		num_boxes = size(boxes, 1);
		data = zeros(4096, num_boxes);
		for kk=1:num_boxes
			patch = im(box(1):box(3), box(2):box(4), :);
			[rep, ~, ~] = caffe_feature(patch);
			data(kk, :) = mean(rep, 2);

			data(kk, :) = sign(data(kk, :)) .* abs(data(kk, :)).^2;
			data(kk, :) = data(kk, :) ./ train_sums;
			data(kk, :) = data(kk, :) ./ sqrt(sum(data(kk, :).^2));
		end

		model = models{ii, 6};
		dec_value = model.w * data';
		abc = [(1:num_boxes)' dec_value'];
		abc = sortrows(abc, 2);
		abc = abc(end:-1:1, :);

		% Get the top positive
		figure;
		imshow(im);
		for kk=1:k
			index = abc(kk, 1);
			box = boxes(index, :);
			rectangle('Position', [box(2) box(1) box(4)-box(2) box(3)-box(1)], 'EdgeColor', 'r');
	    end
	end
end
