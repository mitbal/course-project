% Extract training samples, both positive and negative

setParams;
addpath('/home/iqbal/caffe/matlab/caffe/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/VOCcode/');
addpath('SelectiveSearchCodeIJCV');
addpath('SelectiveSearchCodeIJCV/Dependencies');

% Load all training images
VOC07path = '/home/iqbal/data/VOC07/VOCdevkit/VOC2007/';
imdirpath = [VOC07path 'JPEGImages/'];
train_imgs = textread([VOC07path 'ImageSets/Main/trainval.txt'], '%s');
annopath = [VOC07path 'Annotations/'];
num_train = length(train_imgs);

features = cell(20, 1);
counter = struct('pos', 0, 'neg', 0);
for ii=1:20
	features{ii} = struct('positive', cell(1), 'negative', cell(1));
end

threshold = 0.5;

tic
for ii=1:num_train
	impath = [imdirpath train_imgs{ii} '.jpg'];
	im = imread(impath);
	rec = PASreadrecord([annopath train_imgs{ii} '.xml']);
	
	boxes = selective_search(im);
	num_boxes = size(boxes, 1);	
	num_object = size(rec.objects, 2);

	for jj=1:num_boxes
		ya1 = boxes(jj, 1);
		xa1 = boxes(jj, 2);
		ya2 = boxes(jj, 3);
		xa2 = boxes(jj, 4);

		for kk=1:num_object
			yb1 = rec.objects(kk).bndbox.ymin;
			xb1 = rec.objects(kk).bndbox.xmin;
			yb2 = rec.objects(kk).bndbox.ymax;
			xb2 = rec.objects(kk).bndbox.xmax;

			% Compute overlap
			% First case, no overlap
			if ya1 > yb2 || ya2 < yb1 || xa1 > xb2 || xa2 < xb1
				width = 0; height = 0;
			else 
				if xa2 > xb2
					width = xb2 - xa1;
				else
					width = xa2 - xb1;
				end
				if ya2 > yb2
					height = yb2 - ya1;
				else
					height = ya2 - yb1;
				end
			end

			area_intersect = width * height;
			area_a = (xa2 - xa1) * (ya2 - ya1);
			area_b = (xb2 - xb1) * (yb2 - yb1);
			area_union = area_a + area_b - area_intersect;

			overlap_ratio = double(area_intersect) / area_union;

			IndexC = strfind(params.VOCclasses, rec.class);
			index = find(not(cellfun('isempty', IndexC)));

			patch = im(ya1:ya2, xa1:xa2, :);
			[rep, ~, ~] = caffe_features(patch, params);
			if overlap_ratio > threshold
				% positive sample
				features{index}.positive{counter(index, 1)} = rep;
				counter(index, 1) = counter(index, 1) + 1;
			elseif overlap_ratio < threshold && overlap_ratio > 0
				% neither positive nor negative
			else
				% negative sample
				features{index}.negative{counter(index, 2)} = rep;
				counter(index, 2) = counter(index, 2) + 1;
			end
		end
	end

end

save(['../data/caffe/VOC07-ssfull_' feature_params '.mat'], 'features', '-v7.3');

toc
