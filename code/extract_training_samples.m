%% Extract training samples, both positive and negative

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
counter = zeros(20, 2);
for ii=1:20
	features{ii} = struct('positive', cell(1), 'negative', cell(1));
end

threshold = 0.3;
%%
tic
for ii=1:1
    tic
    disp(['Image: ' num2str(ii)]);
	impath = [imdirpath train_imgs{ii} '.jpg'];
	im = imread(impath);
	rec = PASreadrecord([annopath train_imgs{ii} '.xml']);
	
	boxes = selective_search(im);
	num_boxes = size(boxes, 1);	
	num_object = size(rec.objects, 2);
    disp(['boxes: ' num2str(num_boxes)]);
	for jj=1:num_boxes
		ya1 = boxes(jj, 1);
		xa1 = boxes(jj, 2);
		ya2 = boxes(jj, 3);
		xa2 = boxes(jj, 4);
        
        patch = im(ya1:ya2, xa1:xa2, :);
		%[rep, ~, ~] = caffe_features(patch, params);
        
        % Determine if this regions intersect with any object
		for kk=1:num_object
            obj = rec.objects(kk);
			yb1 = obj.bndbox.ymin;
			xb1 = obj.bndbox.xmin;
			yb2 = obj.bndbox.ymax;
			xb2 = obj.bndbox.xmax;

			% Compute overlap
			% First case, no overlap
            if ya1 > yb2 || ya2 < yb1 || xa1 > xb2 || xa2 < xb1
                width = 0; height = 0;
            else
                if xa2 > xb2
                    if xa1 > xb1
                        width = xb2 - xa1;
                    else
                        width = xb2 - xb1;
                    end
                else
                    if xa1 < xb1
                        width = xa2 - xb1;
                    else
                        width = xa2 - xa1;
                    end
                end
                if ya2 > yb2
                    if ya1 > yb1
                        height = yb2 - ya1;
                    else
                        height = yb2 - yb1;
                    end
                else
                    if ya1 < yb1
                        height = ya2 - yb1;
                    else
                        height = ya2 - ya1;
                    end
                end
            end
            
            area_intersect = width * height;
            area_a = (xa2 - xa1) * (ya2 - ya1);
            area_b = (xb2 - xb1) * (yb2 - yb1);
            area_union = area_a + area_b - area_intersect;
            overlap_ratio = double(area_intersect) / area_union;

            label = rec.objects(kk).class;
			IndexC = strfind(params.VOCclasses, label);
			index = find(not(cellfun('isempty', IndexC)));

			if overlap_ratio > threshold
				% positive sample
                disp(['positive ' label]);
                disp(['box ' num2str(boxes(jj, :)) ' ground truth ' num2str(obj.bbox) ' overlap ' num2str(overlap_ratio)]);
                counter(index, 1) = counter(index, 1) + 1;
				%features{index}.positive{counter(index, 1)} = mean(rep, 2);
                
                imshow(im);
                hold on;
                rectangle('Position', [xb1 yb1 xb2-xb1 yb2-yb1], 'EdgeColor', 'b');
                rectangle('Position', [xa1 ya1 xa2-xa1 ya2-ya1], 'EdgeColor', 'r');
                hold off;
                imshow(patch);
                eye(2);
			elseif overlap_ratio < threshold && overlap_ratio > 0
				% neither positive nor negative
			else
				% negative sample
                counter(index, 2) = counter(index, 2) + 1;
				%features{index}.negative{counter(index, 2)} = mean(rep, 2);
                
                imshow(im);
                hold on;
                rectangle('Position', [xb1 yb1 xb2-xb1 yb2-yb1], 'EdgeColor', 'b');
                rectangle('Position', [xa1 ya1 xa2-xa1 ya2-ya1], 'EdgeColor', 'w');
                hold off;
                imshow(patch);
                eye(2);
			end
		end
    end
    
    toc
end

feature_params = [num2str(params.layerInd) '_' num2str(params.numJitter)];
save(['../data/caffe/VOC07-ssfull_' feature_params '.mat'], 'features', '-v7.3');

toc
