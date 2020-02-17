% This script is designed to take calcium imaging data in form of time
% series of fluorescence intensities and produces heat maps of activities
% as well as sorting of the activity maps

% The input data should be an excel file with each column corresponding to each
% cell of interest and the rows corresponding to intensity profiles


%tmp = xlsread('C:\Users\diwakar.turaga\Desktop\LSFM_data_files\time_trace_24000_agarose_d30_tseries2_xls.xlsx');
tmp = xlsread('E:\Diwakar - McDevitt Lab\11_16_2017\24000_agarose_d30_t_wo_ca_tseries2_rois_circle_excel.xlsx');


traces = tmp(2:end, 2:end);

traces_norm = zeros(size(traces));

show_traces = 1;  
if show_traces
    figure
end

acq_time = 0.056; % 56 msec/frame
total_frames = size(traces,1);
xtime = 0:acq_time:(total_frames-1).* acq_time;

for indx = 1:size(traces,2)
%for indx = 1:5
    %plot(traces(:,indx));
    tmp2 = traces(:,indx);
    traces_norm(:, indx) = (tmp2 - min(tmp2))./min(tmp2);
    %traces_norm(:, indx) = tmp2; % to plot un-normalized
    if show_traces
        
        plot(xtime, traces_norm(:, indx));
        hold on
        %title(['ROI#' num2str(indx)]);
        title(['ROIs'])
        xlabel('seconds');
        %ylabel('normalized intensity');
        ylabel('total intensity');
        %pause  
    end  
end

%legend('1', '2', '3', '4', '5', '6', '7', '8');

traces_norm = traces_norm';

hfig = figure; him = imagesc(traces_norm);
%colormap(hot)
xlabel('seconds')
ylabel('ROI Number')
h = colorbar;
ylabel(h, '\Delta F/F')
title('Unsorted')
set(gca,'Xtick',1:90:499,'XTickLabel',{'0', '5', '10', '15', '20', '25'})

%power spectrum
fs = 18; % cycles per second = 18
[pxx, f] = periodogram(traces_norm',[], [], fs);
pxx2 = pxx';
%pxx2 = log10(pxx2);
max_freq = 50;
figure; 
    for i = 1:size(pxx2, 1); 
        plot(f(2:max_freq), pxx2(i, 2:max_freq)); %arbitrarily chosing 100
        title('Power spectrum')
    ylabel('Units')
    xlabel('Hz')
        hold on; 
        %pause
    end
    
%legend('1', '2', '3', '4', '5', '6', '7', '8');
%set(gca,'Xtick',1:8:max_freq,'XTickLabel',{'0', '1', '2', '3', '4', '5'})

figure; imagesc(pxx2(:, 2:max_freq));
title('Power spectrum')
ylabel('ROI Number')
xlabel('freq')
set(gca,'Xtick',1:5.7971:max_freq,'XTickLabel',{'0', '0.2', '0.4', '0.6', '0.8', '1', '1.2', '1.4', '1.6'})
colorbar

kmeans_sort = 1;
if kmeans_sort

    num_clust = 7;

    idx = kmeans(traces_norm,num_clust);
    %idx = kmeans(pxx2,num_clust);
    [Y,I] = sortrows(idx);
    traces_ordered = traces_norm(I, :); 
    %traces_ordered = pxx2(I, :); 
    max_inten = 0.7;
    traces_ordered(traces_ordered>max_inten) = max_inten; %getting rid of high delta F/F
    figure; imagesc(traces_ordered);
    %colormap(hot)
    xlabel('seconds')
    ylabel('ROI Number')
    title(['Intensities: Sorted with k-means, k = ' num2str(num_clust)]);
    h = colorbar;
    ylabel(h, '\Delta F/F')
    set(gca,'Xtick',1:90:499,'XTickLabel',{'0', '5', '10', '15', '20', '25'})
end

% Now to plot powerspectrum based on same clustering
pxx2_ordered = pxx2(I, :); 
figure; imagesc(pxx2_ordered(:, 2:max_freq));
xlabel('Hz')
ylabel('ROI Number')
title(['Power Spectrum: Sorted with k-means, k = ' num2str(num_clust)]);
h = colorbar;
set(gca,'Xtick',1:5.7971:max_freq,'XTickLabel',{'0', '0.2', '0.4', '0.6', '0.8', '1', '1.2', '1.4', '1.6'})
ylabel(h, 'Units');

figure; hist(Y); 
figure; plot(Y);