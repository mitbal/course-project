%% Compute the average precision, precision, and recall
% of selective search approach to image classification

% Change of approach since memory consumption is of concern
% Train and testing separately to minimize memory

% The second optimization, hard negative mining

setParams;
addpath('/home/iqbal/liblinear-1.93/matlab');

% Load the extracted data
disp(['Load extracted training data...']);
tic
feature_params = [num2str(params.layerInd), '_', num2str(params.numJitter), ...
                    '_', num2str(params.modelItr), '_', num2str(params.modelDataset)];
features_name = ['../data/', params.model, '/', 'VOC07-sstrain', feature_params, '.mat'];
load(features_name);
toc

% Normalization
disp(['Normalization...']);
tic
for i=1:size(trainF, 1)
    trainF(i, :) = sign(trainF(i, :)) .* abs(trainF(i, :)).^2;
end
toc
train_sums = sum(abs(trainF));
tic
for i=1:size(testF, 1)
    testF2(i, :) = testF2(i, :) ./ train_sums;
end
toc
tic
for i=1:size(testF, 1)
    trainF(i, :) = trainF(i, :) ./ sqrt(sum(trainF(i, :).^2));
end
toc

% Change difficult label to the specified class
trainL(trainL == 0) = 1;

num_class = 20;
num_c = 12;

% Penalty parameter for SVM
C = 0.1;
for 1=1:num_c
    Cs(i) = C;
    C = C*2;
end

% Append training label
disp(['Append training label...']);
trainL2 = zeros(size(trainF, 1), num_class);
boxes = cumsum(train_boxes);
for ii=1:size(boxes, 1)
    if ii=1
        start_index = 1;
        end_index = boxes(1);
    else
        start_index = boxes(ii-1);
        end_index = boxes(ii);
    end
    trainL2(start_index:end_index, :) = repmat(trainL(ii, :), [boxes(ii) 1]);
end

% Search the best parameter for SVM
for ci=1:num_c
    for cli=1:num_class
        disp(['Now training data from class: ', num2str(cli), ' and with C: ', num2str(ci)]);
        tic

        % Divide the dataset into positive and negative class
        index = trainL2(:, cli) > 0;
        Sp = trainF(index, :);
        numPos = size(Sp, 1);

        index = trainL2(:, cli) < 0;
        Sn = trainF(index, :);
        numNeg = size(Sn, 1);

        index = randperm(numNeg);
        Snprime = Sn(index(1:numPos, :));

        for ii=1:5
            numNeg = size(Snprime, 1);
            disp(['Iteration ii: ', num2str(ii), ' The number of positive samples: ', num2str(numPos), ' negative: ', num2str(numNeg)]);
            D = [Sp; Snprime];
            sparse_D = sparse(D);
            clear D;

            trainL3 = [ones(numPos,1); -ones(numNeg, 1)];
            model = liblinear_train(trainL3, sparse_D, ['-c ', num2str(Cs(ci)) ' -s 2']);

            % Find hard negative training data
            dec = model.w * Sn' - model.bias;
            index = dec > -1;
            Snprime = [Snprime; Sn(index, :)];
        end

        models{cli, ci} = model;
        toc
    end
end

% Save the model
save(['../models/' feature_params '.mat'], 'models', 'train_sums');
