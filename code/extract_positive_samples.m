% Extract positive samples

setParams;
addpath('/home/iqbal/caffe/matlab/caffe/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/');
addpath('/home/iqbal/data/VOC07/VOCdevkit/VOCcode/');

% Load all training images
VOC07path = '/home/iqbal/data/VOC07/';
imdirpath = [VOC07path, 'VOCdevkit/VOC2007/JPEGImages/'];
trainImgs = textread([VOC07path, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'], '%s');
annopath = [VOC07path, 'VOCdevkit/VOC2007/Annotations/'];
%num_train = length(trainImgs);
num_train = 3;

counter = zeros(20, 1);
features = cell(20, 1);

for ii=1:num_train
    impath = [imdirpath trainImgs{ii} '.jpg'];
	im = imread(impath);
	rec = PASreadrecord([annopath trainImgs{ii} '.xml']);

	num_object = size(rec.objects, 2);
	for jj=1:num_object
		label = rec.objects(jj).class;
		IndexC = strfind(params.VOCclasses, label);
		Index = find(not(cellfun('isempty', IndexC)));

		x = rec.objects(jj).bndbox.xmin;
		y = rec.objects(jj).bndbox.ymin;
		w = rec.objects(jj).bndbox.xmax - x;
		h = rec.objects(jj).bndbox.ymax - y;

		patch = im(y:y+h, x:x+w);

		[rep, ~, ~] = caffe_features(patch, params);
		counter(Index) = counter(Index)+1;
		features{Index, counter(Index)} = rep;
	end
end

save('../data/caffe/positive_samples.mat', 'features');
