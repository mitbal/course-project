setParams;
addpath('/home/iqbal/liblinear-1.93/matlab');

for ii=1:20
	disp(['Now training class: ' num2str(ii)]);
	[model, train_sums] = train_ss(ii);
	for jj=1:size(model, 2)
		models{ii, jj} = model{jj};
	end

	save('../models/caffe/VOC07-ss.mat', 'models', 'train_sums');
end
