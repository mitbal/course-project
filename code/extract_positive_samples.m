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
num_train = length(trainImgs);

labels = cell(1);
features = cell(1);
counter = 1;

tic
for ii=1:num_train
    impath = [imdirpath trainImgs{ii} '.jpg'];
	im = imread(impath);
	rec = PASreadrecord([annopath trainImgs{ii} '.xml']);

	num_object = size(rec.objects, 2);
    disp(['Image: ' num2str(ii)]);
	for jj=1:num_object
		label = rec.objects(jj).class;
		IndexC = strfind(params.VOCclasses, label);
		Index = find(not(cellfun('isempty', IndexC)));
        disp(['  Object: ' num2str(jj) ' class: ' label]);
        
		x1 = rec.objects(jj).bndbox.xmin;
		y1 = rec.objects(jj).bndbox.ymin;
		x2 = rec.objects(jj).bndbox.xmax;
		y2 = rec.objects(jj).bndbox.ymax;

		patch = im(y1:y2, x1:x2, :);

		[rep, ~, ~] = caffe_features(patch, params);

		labels{counter} = Index;
		features{counter} = mean(rep, 2);
        counter = counter + 1;
	end
end
toc

save('../data/caffe/positive_samples.mat', 'features', 'labels', '-v7.3');
