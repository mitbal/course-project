% Train selective search selectively

setParams;
addpath('/home/iqbal/liblinear-1.93/matlab')    

% Load negative data
feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter), '_', num2str(params.modelItr), '_', num2str(params.modelDataset)];
load(['../data/caffe/VOC07-sstrain_' feature_params '.mat']);

train_sums = zeros(20, 4096);

num_c = 8;
num_class = 20;
C = 0.1;
for i=1:num_c
    Cs(i) = C;
    C = C*2;
end
%%
for ii=1:1
    % Load the extracted data
    disp(['Load extracted training data...']);
    tic
    feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter)];
    features_name = ['../data/', params.model, '/', 'VOC07-ssfull_', feature_params, '.mat'];
    load(features_name);
    toc
    
    disp(['Now training class: ' num2str(ii)]);
	num_pos = size(features{ii}.positive, 2);
    if(num_pos > 200000)
        num_pos = 200000;
    end
    Sp = zeros(num_pos, 4096);
    
	for jj=1:num_pos
		Sp(jj, :) = features{ii}.positive{jj};
    end
    
    clear features;
    
    num_neg = num_pos;
    if(num_neg > 200000)
        num_neg = 200000;
    end
    
    index = trainL(:, ii) < 0;
    Sn = trainF(index, :);
    Sn = Sn(1:num_neg, :);
    
    D = [Sp; Sn];
	% Normalize
	disp(['Normalization...']);
	tic
	for i=1:size(D, 1)
	    D(i, :) = sign(D(i, :)) .* abs(D(i, :)).^2;
	end
	toc
	train_sums(ii, :) = sum(abs(D));
	tic
	for i=1:size(D, 1)
	    D(i, :) = D(i, :) ./ train_sums(ii, :);
	end
	toc
	tic
	for i=1:size(D, 1)
	    D(i, :) = D(i, :) ./ sqrt(sum(D(i, :).^2));
	end
	toc

	trainL = [ones(num_pos, 1); -ones(num_neg, 1)];

	for jj = 5:6
		sparse_D = sparse(D);
		%clear trainF;

		model = liblinear_train(trainL, sparse_D, ['-c ', num2str(Cs(jj)) ' -s 2']);
		models{ii, jj} = model;
        clear sparse_D;
        
        % See how many misclassified error on training data
        dec = model.w * D';
        label = dec > 0;
        label = label * 2 -1;
        label = label';
        corr = sum(trainL == label);
        acc = corr / size(trainL, 1)
    end

	feature_params = [num2str(params.layerInd) '_' num2str(params.numJitter) '_' num2str(params.modelItr) '_' params.modelDataset];
	%save(['../models/caffe/VOC07-ss_' feature_params '.mat'], 'models', 'train_sums');
end
