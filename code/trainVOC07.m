function trainVOC07(layerInd, numJitter, model,...
                device, modelItr, modelDataset)
           
    setParams;
            
    if(exist('layerInd','var')), params.layerInd = layerInd; end
    if(exist('numJitter','var')), params.numJitter = numJitter; end
    if(exist('model','var')), params.model = model; end
    if(exist('device','var')), params.device = device; end
    if(exist('modelItr','var')), params.modelItr = modelItr; end
    if(exist('modelDataset','var')), params.modelDataset = modelDataset; end
    if(exist('use_gpu','var')), params.use_gpu = use_gpu; end

    name='VOC07';

    featureParams = [num2str(params.layerInd),'_', num2str(params.numJitter),'_',...
                 num2str(params.modelItr),'_', num2str(params.modelDataset)];
    suffix = [params.model,'/',name,'_Feat_',featureParams];
    featureName = ['../data/',suffix,'.mat'];
    if(~exist(featureName,'file'))
        generateVOC07(params.layerInd, params.numJitter, params.model,...
                params.device, params.modelItr, params.modelDataset);
    end
    
    load(featureName);

    addpath('/home/iqbal/liblinear-1.93/matlab/');

    trainF = sign(trainF) .* abs(trainF).^2;
    sums = sum(abs(trainF));
    trainF = trainF ./ repmat(sums, [size(trainF, 1) 1]);
    trainF = trainF ./ repmat(sqrt(sum(trainF.^2,2)), [1 size(trainF, 2)]);

    testF = sign(testF) .* abs(testF).^2;
    testF = testF ./ repmat(sums, [size(testF, 1) 1]);
    testF = testF ./ repmat(sqrt(sum(testF.^2,2)), [1 size(testF, 2)]);

    trainL(trainL == 0) = 1;
    trainF = sparse(trainF);
    testF = sparse(testF);

    numClasses = 20;
    numBases = 1;
    
    C = 0.1;
    for i = 1 : 12
        Cs(i) = C;
        C = C*2;
    end
    numC = length(Cs);

    apsMax = zeros(numClasses,numC);
    apsSum = zeros(numClasses,numC);

    for ci = 1 : numC
        for cli = 1 : numClasses
            Cs(ci);
            model = liblinear_train(trainL(:,cli), trainF, ['-c ' num2str(Cs(ci)) ' -s 2']);
            models{cli,ci} = model;
            save(['../models/' suffix '.mat'], 'models', 'sums');
            [predicted_label, accuracy, dec_values] = liblinear_predict(testL(:,cli), testF, model,['']);
            if trainL(1, cli) == -1
                dec_values = -dec_values;
            end
                    testL2 = testL(1:numBases:end, cli);
                    scoresSum = dec_values(1:numBases:end,:);
                    for b = 2 : numBases
                            scoresSum = scoresSum + dec_values(b:numBases:end,:);
                    end
                    scoresMax = dec_values(1:numBases:end,:);
                    for b = 2 : numBases
                            scoresMax = max(scoresMax,dec_values(b:numBases:end,:));
                    end
            [recMax{cli,ci},precMax{cli,ci},apsMax(cli, ci)] = PR(scoresMax, testL2, cli, ci)
            [recSum{cli,ci},precSum{cli,ci},apsSum(cli, ci)] = PR(scoresSum, testL2, cli, ci)

        save(['../results/' suffix '.mat'], 'Cs', 'apsMax', 'apsSum', 'recMax', 'recSum', 'precMax', 'precSum');
        end
    end

end
