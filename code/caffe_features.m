function [features,t1,t2] = caffe_features(im, params)

    model_file = ['../models/',params.model,'/',params.modelDataset,...
        '_train_iter_',num2str(params.modelItr),'0000'];

    model_def_file = ...
        ['../defs/',params.model,'/deploy_layer_'...
        ,num2str(params.layerInd),'_Jitter_',num2str(params.numJitter),'.prototxt'];


    if caffe('is_initialized') == 0
        if exist(model_file, 'file') == 0
            error('You need a network model file');
        end
        caffe('init', model_def_file, model_file);
        caffe('set_phase_test');
        caffe('set_device',params.device);
        if params.use_gpu
            caffe('set_mode_gpu');
        else
            caffe('set_mode_cpu');
        end

    end

    % prepare oversampled input
    t1=tic;
    input_data = {prepare_image(im,params.numJitter)};
    t1=toc(t1);

    % do forward pass to get scores
    t2=tic;
    features = caffe('forward', input_data);
    t2=toc(t2);
    features = permute(features{1},[3,4,1,2]);
    

   
end

