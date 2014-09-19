function generateVOC07(layerInd, numJitter, model,...
                device, modelItr, modelDataset)
           
    setParams;
    addpath([params.caffePrf,'matlab/caffe/']);
            
    if(exist('layerInd','var')), params.layerInd = layerInd; end
    if(exist('numJitter','var')), params.numJitter = numJitter; end
    if(exist('model','var')), params.model = model; end
    if(exist('device','var')), params.device = device; end
    if(exist('modelItr','var')), params.modelItr = modelItr; end
    if(exist('modelDataset','var')), params.modelDataset = modelDataset; end
    if(exist('use_gpu','var')), params.use_gpu = use_gpu; end

    name='VOC07';
    
    VOC07Path=[params.dataPrf,name,'/'];
    if(~exist(VOC07Path,'file'))       
        system(['mkdir ',VOC07Path]);
        currentDir=cd(VOC07Path);
        system('wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtrainval_06-Nov-2007.tar');
        system('wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCdevkit_08-Jun-2007.tar');
        system('wget http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/VOCtest_06-Nov-2007.tar');
        system('tar xf VOCdevkit_08-Jun-2007.tar');
        system('tar xf VOCtrainval_06-Nov-2007.tar');        
        system('tar xf VOCtest_06-Nov-2007.tar');        
        cd(currentDir);              
    end
   
    annPath = [VOC07Path,'VOCdevkit/VOC2007/JPEGImages/'];

    trainImgs = textread([VOC07Path,'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'], '%s');
    testImgs = textread([VOC07Path,'VOCdevkit/VOC2007/ImageSets/Main/test.txt'], '%s');
    numTrain = length(trainImgs);
    numTest = length(testImgs);
    numClasses = 20;

    trainF = [];
    trainL = zeros(numTrain, numClasses);
    testF = [];
    testL = zeros(numTest, numClasses);

    for p = 1 : numClasses
        [~, ls] = textread([VOC07Path,'VOCdevkit/VOC2007/ImageSets/Main/' params.VOCclasses{p} '_trainval.txt'], '%s %d');
        trainL(:, p) = ls(:);
        [~, ls] = textread([VOC07Path,'VOCdevkit/VOC2007/ImageSets/Main/' params.VOCclasses{p} '_test.txt'], '%s %d');
        testL(:, p) = ls(:);
    end 

    TT=tic;
    for i=1:length(trainImgs)
        trainImgs{i}=[annPath trainImgs{i} '.jpg'];
    end
    for i=1:length(testImgs)
        testImgs{i}=[annPath testImgs{i} '.jpg'];
    end
    
    trainF = caffe_batch_features(trainImgs, params);

    testF = caffe_batch_features(testImgs, params);
    
    
    featureParams = [num2str(params.layerInd),'_', num2str(params.numJitter),'_',...
                 num2str(params.modelItr),'_', num2str(params.modelDataset)];
    
    save(['../data/',params.model,'/',name,'_Feat_',featureParams,'.mat'], ...
        'trainF', 'testF', 'trainL', 'testL', '-v7.3');

end
