function images = prepare_image(image,numJitter)
    d = load('ilsvrc_2012_mean');
    IMAGE_MEAN = d.image_mean;
    IMAGE_DIM = 340;
    IMAGE_MEAN = imresize(IMAGE_MEAN, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    CROPPED_DIM = 227;
    if(size(image,3)==1)
        image = repmat(image,[1,1,3]);
    end
    image = single(image);
    images = zeros(CROPPED_DIM, CROPPED_DIM, 3, numJitter, 'single');

    rot=[0,20,-20,40,-40];
    for r=0:length(rot-1);
        im = imrotate(image,rot(r+1));
        im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
        im = im(:,:,[3 2 1]) - IMAGE_MEAN;

        % add complete image and its mirror
        images(:,:,:,16*r+1) = permute(imresize(im, [CROPPED_DIM CROPPED_DIM], 'bilinear'), [2 1 3]);
        if(numJitter==16*r+1),return;end
        images(:,:,:,16*r+2) = images(end:-1:1, :, :, 11);
        if(numJitter==16*r+2),return;end
        % 4 corners and their mirrors
        images(:,:,:,16*r+3) = permute(im(1:CROPPED_DIM,end-CROPPED_DIM+1:end, :), [2 1 3]);
        if(numJitter==16*r+3),return;end
        images(:,:,:,16*r+4) = images(end:-1:1, :, :, 3);
        if(numJitter==16*r+4),return;end
        images(:,:,:,16*r+5) = permute(im(end-CROPPED_DIM+1:end,1:CROPPED_DIM, :), [2 1 3]);
        if(numJitter==16*r+5),return;end
        images(:,:,:,16*r+6) = images(end:-1:1, :, :, 5);
        if(numJitter==16*r+6),return;end
        images(:,:,:,16*r+7) = permute(im(end-CROPPED_DIM+1:end,end-CROPPED_DIM+1:end, :), [2 1 3]);
        if(numJitter==16*r+7),return;end
        images(:,:,:,16*r+8) = images(end:-1:1, :, :, 7);
        if(numJitter==16*r+8),return;end
        images(:,:,:,16*r+9) = permute(im(1:CROPPED_DIM,1:CROPPED_DIM, :), [2 1 3]);
        if(numJitter==16*r+9),return;end
        images(:,:,:,16*r+10) = images(end:-1:1, :, :, 1);
        if(numJitter==16*r+10),return;end

        % center and its mirror
        st = ceil((IMAGE_DIM-CROPPED_DIM)/2);       
        images(:,:,:,16*r+11) = permute(im(st:st+CROPPED_DIM-1,st:st+CROPPED_DIM-1, :), [2 1 3]);
        if(numJitter==16*r+11),return;end
        images(:,:,:,16*r+12) = images(end:-1:1, :, :, 9);
        if(numJitter==16*r+12),return;end
        
        imr = imrotate(im, 10);
        st = ceil((size(imr,1)-CROPPED_DIM)/2);
        images(:,:,:,16*r+13) = permute(imr(st:st+CROPPED_DIM-1, st:st+CROPPED_DIM-1, :), [2 1 3]);
        if(numJitter==16*r+13),return;end
        images(:,:,:,16*r+14) = images(end:-1:1, :, :, 13);
        if(numJitter==16*r+14),return;end
        imr = imrotate(im, -10);
        st = ceil((size(imr,1)-CROPPED_DIM)/2);
        images(:,:,:,16*r+15) = permute(imr(st:st+CROPPED_DIM-1, st:st+CROPPED_DIM-1, :), [2 1 3]);
        if(numJitter==16*r+15),return;end
        images(:,:,:,16*r+16) = images(end:-1:1, :, :, 15);
        if(numJitter==16*r+16),return;end
    
    end
end

