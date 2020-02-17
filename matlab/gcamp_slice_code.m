% gcamp slice code

% code used to create supplementary movie 3

addpath(genpath('C:\Users\diwakar.turaga\desktop\'));
im_tmp = imread('C:\Users\diwakar.turaga\Desktop\LSFM_data_files\SAB_stuff\CMaCF_d7_1212_mv_sphere3_b_CMaCF_d7_1212_mv_sphere3_2018-02-11T13-31-53', 'tif');
h = figure; 
xspace = 20;

zt_tscale = 0.12; % 120 msec per frame
zt_zscale = 0.86;% 0.86 um per frame in z

% Actual image in x-axis is from 
im = im_tmp(:, :, :);
im = fliplr(im); %flipping left to right
im(918:944, 1:100, :) = ones(size(im(918:944,1:100, :))); % getting rid of 20 um

%for indx = 211:-10:31
count = 1;
for indx = 211:-1:31
    xdepth = 1055 - round(indx .*715/181);
    im_bar = im;
    im_bar(115:860, xdepth-xspace-15:xdepth-xspace, :) = 256.*ones(size(im(115:860, 1:16, :))); %bar
    im_bar(115:860, xdepth-xspace-15:xdepth-xspace, 1:2) = zeros.*ones(size(im(115:860, 1:16, 1:2))); %bar
    subplot(1,2,2); imshow(im_bar);
    set(gca, 'Position',[0.5 0 0.5 1])

    text(8, 920, '20 \mum', 'color', [1 1 1], 'FontSize', 20);
    
    im_seq_base = 'C:\Users\diwakar.turaga\Desktop\LSFM_data_files\SAB_stuff\CMaCF_d7_1212_mv_sphere3_2\CMaCF_d7_1212_mv_sphere3_v2z';
    im_seq_indx = indx-1;
   
    if im_seq_indx<10
        im_indx = [im_seq_base '00' num2str(im_seq_indx)  '_ORG'];
    elseif im_seq_indx>=10 && im_seq_indx<100
        im_indx = [im_seq_base '0' num2str(im_seq_indx)  '_ORG'];
    else
        im_indx = [im_seq_base num2str(im_seq_indx) '_ORG'];
    end
    im_slice = imread(im_indx, 'tif');
    
    im_slice = im_slice(150:1550, 250:1650); 
    %im_slice = imresize(im_slice, [1000 1000]);
    
    %im_min = 187; % looked at whole stack
    im_min = 200;
    %im_max = 2506 - im_min; % looked at whole stack, scaled
    im_max =3000 - im_min;
    % scaling image manually
    im_slice_sc = (im_slice-im_min).*(65536/im_max); % now its scaled from 1 to max
    
    im_slice_rgb = cat(3, zeros(size(im_slice_sc)), im_slice_sc, zeros(size(im_slice_sc)));
    %im_max = max(max(im_slice(:,:,1))); im_min = min(min(im_slice(:,:,1))); 
    %subplot(1,2,2); imshow(im_slice_rgb); axis image; axis off
    im_slice_rgb = flipud(im_slice_rgb); % flipping
    subplot(1,2,1); image(im_slice_rgb); axis image; axis off;
    set(gca, 'Position',[0 0 0.5 1])
    h.Position = [100 100 1800 900];
    %set(h, 'color','k'); 
    text(100, 1200, ['t = ' num2str(round(count.*zt_tscale, 1)) ' s'], 'color', [1 1 1], 'FontSize', 30);
    text(100, 1270, ['z = ' num2str(round(count.*zt_zscale)) ' \mum'], 'color', [1 1 1], 'FontSize', 30);
    count = count+1;
    
    tmp = getframe(h);
    frame{indx} = tmp.cdata;
    drawnow
    end

%return

 % create the video writer with 5 fps
 %writerObj = VideoWriter('gcamp_sectioning.avi','Uncompressed AVI');
 writerObj = VideoWriter('gcamp_sectioning_time_depth.avi');
 writerObj.FrameRate = 8;
 writeObj.Quality = 100;
 writeObj.LosslessCompression = 1;
 % open the video writer
 open(writerObj);
 % write the frames to the video
 for u=length(frame):-1:1
     writeVideo(writerObj, frame{u});
 end
 % close the writer object
 close(writerObj);
    