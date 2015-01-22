function [models, train_sums] = train_selective_model(cli)
% train selective search model per class
%cli = 15;

    setParams;
    disp(['Load positive training data...']);
	tic
	feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter)];
	features_name = ['../data/', params.model, '/', 'VOC07-sspos_', feature_params, '.mat'];
	load(features_name);
	toc
    
	% Load negative training samples
	disp(['Load negative training data...']);
	tic
	feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter), ...
	                    '_', num2str(params.modelItr), '_', num2str(params.modelDataset)];
	features_name = ['../data/', params.model, '/', 'VOC07-sstrain_', feature_params, '.mat'];
	load(features_name);
	toc
    
    %% Combine positive samples with negative samples, only for
    % normalization purpose
    pos_count = zeros(20,1);
    for ii=1:20
       pos_count(ii) = size(features{ii}.positive, 2);
    end
    pos_total = sum(pos_count);
    posF = zeros(pos_total, 4096);
    counter = 0;
    for ii=1:20
       for jj=1:pos_count(ii)
           counter = counter + 1;
           posF(counter, :) = features{ii}.positive{1, jj}';
       end
    end
    trainF = [posF; trainF];

	% Normalization
	disp(['Normalization...']);
	tic
	for i=1:size(trainF, 1)
	    trainF(i, :) = sign(trainF(i, :)) .* abs(trainF(i, :)).^2;
	end
	toc
	train_sums = sum(abs(trainF));
	tic
	for i=1:size(trainF, 1)
	    trainF(i, :) = trainF(i, :) ./ train_sums;
	end
	toc
	tic
	for i=1:size(trainF, 1)
	    trainF(i, :) = trainF(i, :) ./ sqrt(sum(trainF(i, :).^2));
	end
	toc
    
    % Separate them again
    posF = trainF(1:pos_total, :);
    trainF = trainF(pos_total+1:end, :);

	% Change difficult label to the specified class
	trainL(trainL == 0) = 1;

	% Penalty parameter for SVM
	num_c = 12;
    num_class = 20;
	C = 0.1;
	for i=1:num_c
	    Cs(i) = C;
	    C = C*2;
	end

	%% Append training label
	disp(['Append training label...']);
	trainL2 = zeros(size(trainF, 1), num_class);
	boxes = cumsum(trainBoxes);
	for ii=1:size(boxes, 1)
	    if ii==1
	        start_index = 1;
	        end_index = boxes(1);
	    else
	        start_index = boxes(ii-1)+1;
	        end_index = boxes(ii);
	    end
	    trainL2(start_index:end_index, :) = repmat(trainL(ii, :), [trainBoxes(ii) 1]);
    end
    
    cum_pos = cumsum(pos_count);
    if cli == 1
       start_index = 1
    else
        start_index = cum_pos(cli-1);
    end
    end_index = cum_pos(cli);
    Sp = posF(start_index:end_index, :);
    numPos = pos_count(cli);

    index = trainL2(:, cli) < 0;
    Sn = trainF(index, :);
    clear trainF;

    %% Training's time
    for ci=3:8
        disp(['Now training data from class: ', num2str(cli), ' and with C: ', num2str(ci)]);
        tic

        Snprime = Sn(1:numPos, :);
        Spprime = Sp(1:numPos, :);
        numNeg = size(Snprime, 1);
        disp(['Iteration: 1 ', 'The number of positive samples: ', num2str(numPos), ' negative: ', num2str(numNeg)]);
        D = sparse([Spprime; Snprime]);
        clear Snprime; clear Spprime;

        trainL3 = [ones(numPos,1); -ones(numNeg, 1)];
        model = liblinear_train(trainL3, D, ['-c ', num2str(Cs(ci)) ' -s 2']);
        toc

        % Find hard negative training data
        dec = model.w * Sn';
        index = dec > 0;
        Snprime = [Sn(1:numPos, :); Sn(index, :)];
        
        % Limit the total number of samples to 400,000 due to memory
        % constraint
        sisa = 400000 - numPos;
        if size(Snprime, 1) > sisa
            Snprime = Snprime(1:sisa, :);
        end
        Spprime = Sp(1:numPos, :);

        % Retrain using hard negative training data
        D = sparse([Spprime; Snprime]);
        numNeg = size(Snprime, 1);
        clear Snprime; clear Spprime;

        tic
        disp(['Iteration: 2 ', 'The number of positive samples: ', num2str(numPos), ' negative: ', num2str(numNeg)]);
        trainL3 = [ones(numPos, 1); -ones(numNeg, 1)];
        model = liblinear_train(trainL3, D, ['-c ', num2str(Cs(ci)) ' -s 2']);
        clear D;
        toc

        % Find out how many misclassified negative data
        dec = model.w * Sn';
        index = dec > 0;
        sum(index)

        models{ci} = model;
        toc
    end
end
