% xyz slice code

% Code to create supplementary movie 1

%addpath(genpath('C:\Users\diwakar.turaga\desktop\'));
im = imread('D:\McDevitt_lab\LSFM_data_files\xyz_slice_movie\f4_4color_sphere2_mv_filebasedregistration_C0_Z000_2017-10-27T15-25-26', 'tif');

h = figure; 
for indx = 1:44
    xdepth = round(indx .*480/22);
    im_bar = im;
    im_bar(50:950, xdepth:xdepth+15, :) = 256.*ones(size(im(50:950, xdepth+15:xdepth+30, :))); %bar
    subplot(1,2,1); imshow(im_bar);

    im_seq_base = 'D:\McDevitt_lab\LSFM_data_files\xyz_slice_movie\xyz_slice\f4_4color_sphere2_mv_filebasedregistration_C3_Z0';
    im_seq_indx = indx-1;
    if im_seq_indx<10
        im_indx = [im_seq_base num2str(0) num2str(im_seq_indx)];
    else
        im_indx = [im_seq_base num2str(im_seq_indx)];
    end
    im_slice = imread(im_indx, 'tif');
    %im_min = 187; % looked at whole stack
    im_min = 400;
    %im_max = 2506 - im_min; % looked at whole stack, scaled
    im_max = 2200 - im_min;
    % scaling image manually
    im_slice_sc = (im_slice-im_min).*(65536/im_max); % now its scaled from 1 to max
    
    im_slice_rgb = cat(3, im_slice_sc, zeros(size(im_slice_sc)), zeros(size(im_slice_sc)));
    %im_max = max(max(im_slice(:,:,1))); im_min = min(min(im_slice(:,:,1))); 
    %subplot(1,2,2); imshow(im_slice_rgb); axis image; axis off
    subplot(1,2,2); image(im_slice_rgb); axis image; axis off
    h.Position = [50 50 1800 900];
    set(h, 'color','k'); 
    tmp = getframe(h);
    frame{indx} = tmp.cdata;
    drawnow
end

 % create the video writer with 5 fps
 writerObj = VideoWriter('xyz_sectioning.avi','Uncompressed AVI');
 %writerObj = VideoWriter('xyz_sectioning_small.avi');
 writerObj.FrameRate = 5;
 writeObj.Quality = 100;
 writeObj.LosslessCompression = 1;
 % open the video writer
 open(writerObj);
 % write the frames to the video
 for u=1:length(frame)
     writeVideo(writerObj, frame{u});
 end
 % close the writer object
 close(writerObj);
