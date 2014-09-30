setParams;
addpath('/home/iqbal/liblinear-1.93/matlab');
% The same function with train_selective_search but now the
% training is using data without region proposal (standard)

% Load the extracted data
feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter), ...
					'_', num2str(params.modelItr), '_', num2str(params.modelDataset)];
features_name = ['../data/', params.model, '/', 'VOC07-ss', feature_params, '.mat'];
load(features_name);

% Load pre-trained model
model_name = ['../models/caffe/', 'VOC07_Feat_', feature_params, '.mat'];
load(model_name);

% Normalization
testF = sign(testF) .* abs(testF).^2;
testF = testF ./ repmat(sums, [size(testF, 1) 1]);
testF = testF ./ repmat(sqrt(sum(testF.^2, 2)), [1 size(testF, 2)]);

% Change to sparse matrix because the liblinear demand so
testF = sparse(testF);

num_c = 12; % Number of different penalty parameter
num_classes = 20; % Number of classes in VOC dataset

c = 0.1;
for i=1:num_c
	Cs(i) = c;
	c = c*2;
end

% It's prediction time
for ci=1:num_c
	for cli=1:num_classes

		testL2 = zeros(size(testF, 1), 1);
		model = models{cli, ci};
		[predicted_labels, accuracy, dec_value] = liblinear_predict(testL2, testF, model, ['']);

		% Aggregate the result. There are two approaches, the maximum value and the average.
		num_imgs = size(testL, 1);
		%num_imgs = 10;
		scores_sum = zeros(num_imgs, size(dec_value, 2));
		scores_max = zeros(num_imgs, size(dec_value, 2));
		boxes = cumsum(testBoxes);
		for ii=1:num_imgs
			if ii == 1
				scores_sum(ii, :) = sum(dec_value(1:boxes(ii), :));
				scores_max(ii, :) = max(dec_value(1:boxes(ii), :));
			else
				scores_sum(ii, :) = sum(dec_value(boxes(ii-1)+1:boxes(ii), :));
				scores_max(ii, :) = max(dec_value(boxes(ii-1)+1:boxes(ii), :));
			end
		end

		% Calculate precision, recall, and average precision
		[rec_max{cli, ci}, prec_max{cli, ci}, aps_max(cli, ci)] = PR(scores_max, testL(:, cli), cli, ci);
		[rec_sum{cli, ci}, prec_sum{cli, ci}, aps_sum(cli, ci)] = PR(scores_sum, testL(:, cli), cli, ci)
	end
end

% Save the result to file
save(['../results/', params.model, '/', 'VOC07-ss', feature_params, '.mat'], 'Cs', 'aps_max', 'aps_sum', 'rec_max', 'rec_sum', 'prec_max', 'prec_sum');
