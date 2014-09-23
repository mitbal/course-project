function [boxes, features] = extract_feature_all_boxes(imgs,params)
    
    numImgs = length(imgs);
    boxes = {numImgs};
    features = {};
    counter = 1;
    for i = 1:numImgs
        im = imread(imgs{i});

        disp(['selective search to image ', num2str(i)]);
		tic
        boxes{i} = selective_search(im);
		toc
        for j=1:length(boxes{i})
            box = boxes{i}(1, :);
            x = box(1); y = box(2); w = box(3); h = box(4);
            patch = im(x:w, y:h);
            [rep, ~, ~] = caffe_features(patch, params);
            features{counter} = mean(rep, 2);
            counter = counter + 1;
        end        
    end
end

