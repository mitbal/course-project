% Main script to produce baseline result of image classification of PASCAL VOC 2007 dataset


%% Experiment 1
layer_index = 7; %the second fully-connected layer
num_jitter = 16; %number of jittered images
model = 'caffe';
device = 0; %first look at available GPU using nvidia-smi
model_itr = 0;
model_dataset = 'imagenet';
use_gpu = 1;

% Download, if necessary, and then extracted the CNN features of training and testing images
generateVOC07(layer_index, num_jitter, model, device, model_itr, model_dataset);

% Train SVM from extracted CNN features
trainVOC07(layer_index, num_jitter, model, device, model_itr, model_dataset);

% The result then can be looked from .mat file at results directory by looking at apsMax variable column with higher mean average precision (mAP)


%% Experiment 2
layer_index = 6; %the first fully-connected layer
num_jitter = 16;
model = 'caffe';
device = 0;
model_itr = 0;
model_dataset = 'imagenet';
use_gpu = 1;

generateVOC07(layer_index, num_jitter, model, device, model_itr, model_dataset);

trainVOC07(layer_index, num_jitter, model, device, model_itr, model_dataset);


%% Experiment 3
layer_index = 5; %the last convolution layer, after last relu unit
num_jitter = 16;
model = 'caffe';
device = 0;
model_itr = 0;
model_dataset = 'imagenet';
use_gpu = 1;

generateVOC07(layer_index, num_jitter, model, device, model_itr, model_dataset);

trainVOC07(layer_index, num_jitter, model, device, model_itr, model_dataset);

