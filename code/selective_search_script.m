% Main script to run selective search for classification

% set Parameters
setParams;
addpath('SelectiveSearchCodeIJCV');
addpath('SelectiveSearchCodeIJCV/Dependencies');

% Load all VOCImage
im = imread('000015.jpg');
boxes = selective_search(im);

