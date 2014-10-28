% Set global parameter for each caffe invocation

global params

params.dataPrf = '/home/iqbal/data/';	% Location to save the dataset
params.caffePrf = '/home/iqbal/caffe/';	% Location to caffe installation

% Default parameter is the function is called without parameter
params.layerInd = 7; 
params.numJitter = 16; 
params.model = 'caffe'; 
params.device = 0; 
params.modelItr = 0;
params.modelDataset = 'imagenet';
params.use_gpu=1;

% 20 classes in VOC dataset
params.VOCclasses =  {'aeroplane' 'bicycle' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair' 'cow' 'diningtable' 'dog' 'horse' 'motorbike' 'person' 'pottedplant' 'sheep' 'sofa' 'train' 'tvmonitor'};
params.VOCaclasses = {'jumping', 'playinginstrument' 'ridingbike' 'running', 'usingcomputer' 'phoning' 'reading' 'ridinghorse' 'takingphoto' 'walking'};
