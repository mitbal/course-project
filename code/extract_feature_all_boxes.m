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
		count_boxes(i) = length(boxes{i});
		disp(['number of regions ', num2str(count_boxes(i))]);
        for j=1:length(boxes{i})
            box = boxes{i}(j, :);
            x = box(1); y = box(2); w = box(3); h = box(4);
            patch = im(x:w, y:h, :);
            [rep, ~, ~] = caffe_features(patch, params);
            features{counter} = mean(rep, 2);
            counter = counter + 1;
        end
		toc
    end
end

