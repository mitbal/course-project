function features = caffe_batch_features(imgs,params)
 
    features = [];
    
    TT=tic;
    numImgs = length(imgs);
    for i = 1 : numImgs
        tr(i)=tic;
        im = imread(imgs{i});
        tr(i)=toc(tr(i));
        [rep,tp(i),te(i)] = caffe_features(im, params);
        disp(['Imgs{' num2str(i),'}',...
            ' Read:',num2str(1000*tr(i),'%4.2f'),'ms',...
            ' Prepare:',num2str(1000*tp(i),'%4.2f'),'ms',...
            ' Extract:',num2str(1000*te(i),'%4.2f'),'ms']);                        
        if(isempty(features))
            features = zeros(numImgs,size(rep,1));
        end
        features(i, :) = mean(rep,2);
    end
    TT = toc(TT);
    TRtr=mean(tr);
    TEtr=mean(te);
    TPtr=mean(tp);
    disp(['Feature extracted in ', num2str(TT,'%8.2f'), ' Sec,',...
        ' mean read: ', num2str(1000*TRtr,'%8.2f'), 'ms,',...
        ' mean prepare: ', num2str(1000*TPtr,'%8.2f'), 'ms,',...
        ' mean extract: ', num2str(1000*TEtr,'%8.2f'), 'ms,',...
        ' average overload time: ', num2str(1000*((TT-numImgs*(TRtr+TEtr+TPtr))/numImgs),'%8.3f'), 'ms']);
    
end