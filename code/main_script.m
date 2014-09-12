% Main script to produce baseline result of image classification of PASCAL VOC 2007 dataset

layer_index = 7; %the second fully-connected layer
num_jitter = 16; %still don't understand
model = 'caffe';
device = 0;
model_itr = 0;
model_dataset = 'imagenet';
use_gpu = 1;

% Download, if necessary, and then extracted the CNN features of training and testing images
generateVOC2007(layer_index, num_jitter, model, device, model_itr, model_dataset);

