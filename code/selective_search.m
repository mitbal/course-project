function [boxes] = selective_search(im)
% Function selective_search return region proposal which is indicated to have object in it based on segmentation result

	% set color space to be used
	colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
	colorTypes = colorTypes{1};

	% set similarity measure
	simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
	simFunctionHandles = simFunctionHandles(1);

	% parameter for Felzenszwalb segmentation algorithm
	k = 100;
	minSize = k;
	sigma = 0.8;

	% perform selective search
	[boxes, ~, ~, ~] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorTypes, simFunctionHandles);
	boxes = BoxRemoveDuplicates(boxes);

end
