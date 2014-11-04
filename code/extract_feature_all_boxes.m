function [count_boxes, features] = extract_feature_all_boxes(imgs,params)
    
    numImgs = length(imgs);
    boxes = {numImgs};
    count_boxes = zeros(numImgs, 1);
    features = {};
    counter = 1;
    disp(['Selective Search']);
    for i = 1:numImgs
        im = imread(imgs{i});

        disp(['Image: ', num2str(i)]);
		tic
        boxes{i} = selective_search(im);
		count_boxes(i) = size(boxes{i}, 1);
		disp(['number of regions ', num2str(count_boxes(i))]);
        for j=1:size(boxes{i}, 1)
            box = boxes{i}(j, :);
            y1 = box(1); x1 = box(2); y2 = box(3); x2 = box(4);
            patch = im(y1:y2, x1:x2, :);
            [rep, ~, ~] = caffe_features(patch, params);
            features{counter} = mean(rep, 2);
            counter = counter + 1;
        end
		toc
    end
end

