% Train selective search selectively

setParams;
% Load the extracted data
disp(['Load extracted training data...']);
tic
feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter)];
features_name = ['../data/', params.model, '/', 'VOC07-ssfull_', feature_params, '.mat'];
load(features_name);
toc

train_sums = zeros(20, 1);

num_c = 8;
num_class = 20;
C = 0.1;
for i=1:num_c
    Cs(i) = C;
    C = C*2;
end

for ii=1:20
	num_pos = size(features{ii}.positive, 1);
	num_neg = size(features{ii}.negative, 1);
	trainF = zeros(num_pos+num_neg, 4096);
	for jj=1:num_pos
		trainF(jj, :) = features{ii}.positive{jj};
	end
	for jj=num_pos+1:num_pos+num_neg
		trainF(jj, :) = features{ii}.negative{jj};
	end
	
	% Normalize
	disp(['Normalization...']);
	tic
	for i=1:size(trainF, 1)
	    trainF(i, :) = sign(trainF(i, :)) .* abs(trainF(i, :)).^2;
	end
	toc
	train_sums(ii) = sum(abs(trainF));
	tic
	for i=1:size(trainF, 1)
	    trainF(i, :) = trainF(i, :) ./ train_sums(ii);
	end
	toc
	tic
	for i=1:size(trainF, 1)
	    trainF(i, :) = trainF(i, :) ./ sqrt(sum(trainF(i, :).^2));
	end
	toc

	trainL = [ones(num_pos, 1); -ones(num_neg, 1)];

	for jj = 6:6
		sparse_D = sparse(trainF);
		clear trainF;

		model = liblinear_train(trainL, sparse_D, ['-c ', num2str(Cs(jj)) ' -s 2']);
		models{ii, ci} = model;
	end

	feature_params = [num2str(params.layerInd) '_' num2str(params.numJitter) '_' num2str(params.modelItr) '_' params.modelDataset];
	save(['../data/models/VOC07-ss_' feature_params '.mat'], 'models', 'train_sums');
end
