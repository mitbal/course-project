%% Visualize top k bounding box in positive image
close all; clear all; clc;

setParams;
addpath('/home/iqbal/caffe/matlab/caffe/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/VOCcode/');
addpath('SelectiveSearchCodeIJCV/');
addpath('SelectiveSearchCodeIJCV/Dependencies/');

% Load all training images and necessary models
num_class = 20;
VOC07path = '/home/iqbal/data/VOC07/';
imdirpath = [VOC07path, 'VOCdevkit/VOC2007/JPEGImages/'];
train_imgs = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'], '%s');
annopath = [VOC07path, 'VOCdevkit/VOC2007/Annotations/'];
num_train = length(train_imgs);
trainL = zeros(num_train, 20);
for p = 1 : num_class
    [~, ls] = textread([VOC07path,'VOCdevkit/VOC2007/ImageSets/Main/' params.VOCclasses{p} '_trainval.txt'], '%s %d');
    trainL(:, p) = ls(:);
end

load('../models/caffe/VOC07-ss.mat');

k = 3;
n = 20;

for ii=1:20
    disp(['Processing class: ' num2str(ii)]);
    n_count = 0;
	for jj=1:num_train
        if trainL(jj, ii) < 1
            continue;
        end
        n_count = n_count + 1;
        if n_count > n
            break;
        end
        disp(['Image: ' num2str(jj)]);
		impath = [imdirpath train_imgs{jj} '.jpg'];
		im = imread(impath);

		boxes = selective_search(im);
		num_boxes = size(boxes, 1);
		data = zeros(num_boxes, 4096);
		for kk=1:num_boxes
            box = boxes(kk, :);
			patch = im(box(1):box(3), box(2):box(4), :);
			[rep, ~, ~] = caffe_features(patch, params);
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
        text_str = cell(k, 1);
        position = zeros(k, 2);
        box_color = cell(k, 1);
		for kk=1:k
			index = abc(kk, 1);
			box = boxes(index, :);
            text_str{kk} = num2str(abc(kk, 2), '%0.2f');
            position(kk, :) = [box(2) box(1)];
            box_color{kk} = 'red';
        end
        RGB = insertText(im, position, text_str, 'TextColor', 'white', 'FontSize', 12, 'BoxColor', box_color, 'BoxOpacity', 1);
        h = figure;
        imshow(RGB);
        for kk=1:k
            index = abc(kk, 1);
			box = boxes(index, :);
            rectangle('Position', [box(2) box(1) box(4)-box(2) box(3)-box(1)], 'EdgeColor', 'r');
        end
        saveTightFigure(h, ['../plot/class_' num2str(ii) '_' num2str(jj) '.png']);
        close all;
	end
end
