% Main script to produce baseline result of image classification of PASCAL VOC 2007 dataset

layer_index = 7; %the second fully-connected layer
num_jitter = 16; %number of jittered images
model = 'caffe';
device = 2;
model_itr = 0;
model_dataset = 'imagenet';
use_gpu = 1;

% Download, if necessary, and then extracted the CNN features of training and testing images
generateVOC07(layer_index, num_jitter, model, device, model_itr, model_dataset);

% Train SVM from extracted CNN features
trainVOC07(layer_index, num_jitter, model, device, model_itr, model_dataset);

% The result then can be looked from output images in results directory, by loading it into matlab and look for column in apsMax with overall better value

