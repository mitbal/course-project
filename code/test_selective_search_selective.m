% Test selective search selectively
setParams;

% Load selective search model
feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter), ...
						'_', num2str(params.modelItr), '_', num2str(params.modelDataset)];
model_name = ['../models/caffe/', 'VOC07-selective_', feature_params, '.mat'];
load(model_name);

num_c = 6;
c = 0.1;
for i=1:num_c
	Cs(i) = c;
	c = c*2;
end

for ii=1:1
	% Load extracted testing data
	disp(['Load extracted data']);
	tic
	
	features_name = ['../data/', params.model, '/', 'VOC07-sstest_', feature_params, '.mat'];
	load(features_name);
	toc

	% Normalize
	disp(['Normalization...']);
	tic
	for i=1:size(testF, 1)
	    testF(i, :) = sign(testF(i, :)) .* abs(testF(i, :)).^2;
	end
	toc
	tic
	for i=1:size(testF, 1)
	    testF(i, :) = testF(i, :) ./ train_sums(ii, :);
	end
	toc
	tic
	for i=1:size(testF, 1)
	    testF(i, :) = testF(i, :) ./ sqrt(sum(testF(i, :).^2));
	end
	toc
%%
	for jj=6:6
		tic
		model = models{ii, jj};
		disp(['Prediction for class: ', num2str(ii), ' and C: ', num2str(jj)]);
		dec_value = [];
		div = 5000;
		num_iter = uint32(size(testF, 1) / div);
		for kk=1:num_iter
			start_index = (kk-1)*div + 1;
			end_index = kk*div;
			if end_index > size(testF, 1)
				end_index = size(testF, 1);
			end

			dec = model.w * testF(start_index:end_index, :)';
			dec_value = [dec_value; dec'];
		end
		toc

		% Aggregate the result. There are two approaches, the maximum value and the mean.
		disp(['Aggregate result...']);
		tic
		num_imgs = size(testL, 1);
		scores_sum = zeros(num_imgs, 1);
		scores_max = zeros(num_imgs, 1);
		boxes = cumsum(testBoxes);
		for kk=1:num_imgs
			if kk == 1
				scores_sum(kk) = mean(dec_value(1:boxes(kk)));
				scores_max(kk) = max(dec_value(1:boxes(kk)));
			else
				scores_sum(kk) = mean(dec_value(boxes(kk-1)+1:boxes(kk)));
				scores_max(kk) = max(dec_value(boxes(kk-1)+1:boxes(kk)));
			end
        end
        
        label = scores_max > 0;
        label = label * 2 -1;
        corr = sum(testL(:, ii) == label);
        acc = corr / size(testL, 1)

		% Calculate precision, recall, and average precision
		disp(['Calculate precision, recall, and average precision...']);
		[rec_max{ii, jj}, prec_max{ii, jj}, aps_max(ii, jj)] = PR(scores_max, testL(:, ii), ii, jj)
		[rec_sum{ii, jj}, prec_sum{ii, jj}, aps_sum(ii, jj)] = PR(scores_sum, testL(:, ii), ii, jj);
		toc

		% Save the result to file
		disp(['Saving...']);
		save(['../results/', params.model, '/', 'VOC07-selective_', feature_params, '.mat'], 'Cs', 'aps_max', 'aps_sum', 'rec_max', 'rec_sum', 'prec_max', 'prec_sum');
	end
	clear testF;
end
