setParams;
addpath('/home/iqbal/liblinear-1.93/matlab');
% The same function with train_selective_search but now the
% training is using data without region proposal (standard)
% This is the second optimized code by using simple dot product to perform the prediction

% Load the extracted data
disp(['Load extracted data']);
tic
feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter), ...
					'_', num2str(params.modelItr), '_', num2str(params.modelDataset)];
features_name = ['../data/', params.model, '/', 'VOC07-ss', feature_params, '.mat'];
load(features_name);
toc

% Load to get training data label due to liblinear bug
load(['../data/', params.model, '/', 'VOC07_Feat_', feature_params, '.mat']);

% Load pre-trained model
disp(['Load pre-trained model']);
tic
model_name = ['../models/caffe/', 'VOC07_Feat_', feature_params, '.mat'];
load(model_name);
toc

% Normalization
disp(['Normalization...']);
tic
for i=1:size(testF, 1)
	testF(i, :) = sign(testF(i, :)) .* abs(testF(i, :)).^2;
end
toc
tic
for i=1:size(testF, 1)
	testF(i, :) = testF(i, :) ./ sums;
end
toc
tic
for i=1:size(testF, 1)
	testF(i, :) = testF(i, :) ./ sqrt(sum(testF(i, :).^2));
end
toc

num_c = 10; % Number of different penalty parameter
num_classes = 20; % Number of classes in VOC dataset

c = 0.1;
for i=1:num_c
	Cs(i) = c;
	c = c*2;
end

% It's prediction time
for ci=1:num_c
	for cli=1:num_classes

		tic
		model = models{cli, ci};
		disp(['Prediction for class: ', num2str(cli), ' and C: ', num2str(ci)]);
		dec_value = [];
		div = 5000;
		num_iter = uint32(size(testF, 1) / div);
		for ii=1:num_iter
			start_index = (ii-1)*div + 1;
			end_index = ii*div;
			if end_index > size(testF, 1)
				end_index = size(testF, 1);
			end

			dec = model.w * testF(start_index:end_index, :)' - model.bias;
			dec_value = [dec_value; dec'];
		end
		toc

		% Due to bug in liblinear, the first label in the training data is always
		% taken as class 1
		if trainL(1, cli) == -1
            dec_value = -dec_value;
        end

		% Aggregate the result. There are two approaches, the maximum value and the average.
		disp(['Aggregate result...']);
		tic
		num_imgs = size(testL, 1);
		scores_sum = zeros(num_imgs, 1);
		scores_max = zeros(num_imgs, 1);
		boxes = cumsum(testBoxes);
		for ii=1:num_imgs
			if ii == 1
				scores_sum(ii) = sum(dec_value(1:boxes(ii)));
				scores_max(ii) = max(dec_value(1:boxes(ii)));
			else
				scores_sum(ii) = sum(dec_value(boxes(ii-1)+1:boxes(ii)));
				scores_max(ii) = max(dec_value(boxes(ii-1)+1:boxes(ii)));
			end
		end

		% Calculate precision, recall, and average precision
		disp(['Calculate precision, recall, and average precision...']);
		[rec_max{cli, ci}, prec_max{cli, ci}, aps_max(cli, ci)] = PR(scores_max, testL(:, cli), cli, ci);
		[rec_sum{cli, ci}, prec_sum{cli, ci}, aps_sum(cli, ci)] = PR(scores_sum, testL(:, cli), cli, ci)
		toc

		% Save the result to file
		disp(['Saving...']);
		save(['../results/', params.model, '/', 'VOC07-ss', feature_params, '.mat'], 'Cs', 'aps_max', 'aps_sum', 'rec_max', 'rec_sum', 'prec_max', 'prec_sum');
	end
end
