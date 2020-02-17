% Script to look at total sizes of each of the aggregates and make
% schematic of interested cells

% The data is stored in Imaris format and individual data points saves in
% an excel file

%data filenames
filename_unique{1} = 'a4_4color_sphere1_CF';
filename_unique{2} = 'a4_4color_sphere1_CM';
filename_unique{3} = 'a4_4color_sphere2_CF';
filename_unique{4} = 'a4_4color_sphere2_CM';
filename_unique{5} = 'A4_4color_sphere3_CF';
filename_unique{6} = 'A4_4color_sphere3_CM';
filename_unique{7} = 'A4_4color_sphere4_CF';
filename_unique{8} = 'A4_4color_sphere4_CM';
filename_unique{9} = 'a4_4color_sphere5_CF';
filename_unique{10} = 'a4_4color_sphere5_CM';
filename_unique{11} = 'a4_4color_sphere6_CF';
filename_unique{12} = 'a4_4color_sphere6_CM';
filename_unique{13} = 'a4_4color_sphere7_CF';
filename_unique{14} = 'a4_4color_sphere7_CM';
filename_unique{15} = 'a4_4color_sphere8_CF';
filename_unique{16} = 'a4_4color_sphere8_CM';

filename_begin = 'D:\McDevitt_lab\LSFM_data_files\';
filename_mid1 = '_Statistics\';
filename_end = '_Position.csv';

for i = 1:length(filename_unique)
    filename{i} = [filename_begin filename_unique{i} filename_mid1 filename_unique{i} filename_end];
end

% aCF data analysis
threshold = 300; %assuming largest cell to cell distance is 300 um

count = 1; 
for i = 1:2:length(filename)

    posCM = xlsread(filename{i+1});%CM is usually the second file
    posCF = xlsread(filename{i});%CF is the first file

    offbyone = 0; % for CM-CM only

    % posB = posA; 
    % offbyone = 1;

    [idx, distCMCF] = rangesearch(posCM(:, 1:3),posCF(:, 1:3),threshold);
    
    posCF = posCM; 

    [idx, distCMCM] = rangesearch(posCM(:, 1:3),posCF(:, 1:3),threshold);
    
    dist_CMaCF = [distCMCM; distCMCF];
    dist = cell2mat(dist_CMaCF);
    dist = dist(:);
    dist_sort = sort(dist);
    diameter_CMaCF(count) = mean(dist_sort(end-2:end));
    diameter_CMaCF = round(diameter_CMaCF);
    
    [mesh_CMaCF{count} , vol_CMaCF(count)] = boundary([posCM(:, 1:3) ; posCF(:, 1:3)]);
    
    count = count+1;
    
end


%data filenames FCF
filename_unique{1} = 'f4_4color_sphere1_CF';
filename_unique{2} = 'f4_4color_sphere1_CM';
filename_unique{3} = 'f4_4color_sphere2_CF';
filename_unique{4} = 'f4_4color_sphere2_CM';
filename_unique{5} = 'F4_4color_sphere4_CF';
filename_unique{6} = 'F4_4color_sphere4_CM';
filename_unique{7} = 'f4_4color_sphere5_CF';
filename_unique{8} = 'f4_4color_sphere5_CM';
filename_unique{9} = 'f4_4color_sphere6_CF';
filename_unique{10} = 'f4_4color_sphere6_CM';
filename_unique{11} = 'f4_4color_sphere7_CF';
filename_unique{12} = 'f4_4color_sphere7_CM';
filename_unique{13} = 'f4_4color_sphere31_CF';
filename_unique{14} = 'f4_4color_sphere31_CM';
filename_unique{15} = 'f4_4color_sphere32_CF';
filename_unique{16} = 'f4_4color_sphere32_CM';

%filename_begin = 'C:\Users\diwakar.turaga\Desktop\LSFM_data_files\';
filename_mid1 = '_Statistics\';
filename_end = '_Position.csv';

for i = 1:length(filename_unique)
    filename{i} = [filename_begin filename_unique{i} filename_mid1 filename_unique{i} filename_end];
end

count = 1; 
for i = 1:2:length(filename)

    posCM = xlsread(filename{i+1});%CM is usually the second file
    posCF = xlsread(filename{i});%CF is the first file

    offbyone = 0; % for CM-CM only

    % posB = posA; 
    % offbyone = 1;

    [idx, distCMCF] = rangesearch(posCM(:, 1:3),posCF(:, 1:3),threshold);
    
    posCF = posCM; 

    [idx, distCMCM] = rangesearch(posCM(:, 1:3),posCF(:, 1:3),threshold);
    
    dist_CMfCF = [distCMCM; distCMCF];
    dist = cell2mat(dist_CMfCF);
    dist = dist(:);
    dist_sort = sort(dist);
    diameter_CMfCF(count) = mean(dist_sort(end-2:end));
    diameter_CMfCF = round(diameter_CMfCF);
    
    [mesh_CMfCF{count} , vol_CMfCF(count)] = boundary([posCM(:, 1:3) ; posCF(:, 1:3)]);
    
    count = count+1;
    
end

%data filenames  % CM only
filename_unique{1} = 'c4_4color_sphere2_CF';
filename_unique{2} = 'c4_4color_sphere2_CM';
filename_unique{3} = 'c4_4color_sphere3_CF';
filename_unique{4} = 'c4_4color_sphere3_CM';
filename_unique{5} = 'c4_4color_sphere4_CF';
filename_unique{6} = 'c4_4color_sphere4_CM';
filename_unique{7} = 'c4_4color_sphere5_CF';
filename_unique{8} = 'c4_4color_sphere5_CM';
filename_unique{9} = 'c4_4color_sphere11_CF';
filename_unique{10} = 'c4_4color_sphere11_CM';
filename_unique{11} = 'c4_4color_sphere12_CF';
filename_unique{12} = 'c4_4color_sphere12_CM';
filename_unique{13} = 'c4_4color_sphere13_CF';
filename_unique{14} = 'c4_4color_sphere13_CM';
filename_unique{15} = 'c4_4color_sphere71_CF';
filename_unique{16} = 'c4_4color_sphere71_CM';
filename_unique{17} = 'c4_4color_sphere72_CF';
filename_unique{18} = 'c4_4color_sphere72_CM';

%filename_begin = 'C:\Users\diwakar.turaga\Desktop\LSFM_data_files\';
filename_mid1 = '_Statistics\';
filename_end = '_Position.csv';

for i = 1:length(filename_unique)
    filename{i} = [filename_begin filename_unique{i} filename_mid1 filename_unique{i} filename_end];
end

count = 1; 
for i = 1:2:length(filename)

    posCM = xlsread(filename{i+1});%CM is usually the second file
    posCF = xlsread(filename{i});%CF is the first file

    offbyone = 0; % for CM-CM only

    % posB = posA; 
    % offbyone = 1;

    [idx, distCMCF] = rangesearch(posCM(:, 1:3),posCF(:, 1:3),threshold);
    
    posCF = posCM; 

    [idx, distCMCM] = rangesearch(posCM(:, 1:3),posCF(:, 1:3),threshold);
    
    dist_CMoCF = [distCMCM; distCMCF];
    dist = cell2mat(dist_CMoCF);
    dist = dist(:);
    dist_sort = sort(dist);
    diameter_CMoCF(count) = mean(dist_sort(end-2:end));
    diameter_CMoCF = round(diameter_CMoCF);
    
    [mesh_CMoCF{count} , vol_CMoCF(count)] = boundary([posCM(:, 1:3) ; posCF(:, 1:3)]);
    
    count = count+1;
    
end

return

figure;
diameter_all = [diameter_CMaCF diameter_CMfCF diameter_CMoCF];
g = [zeros(1, length(diameter_CMaCF)) ones(1, length(diameter_CMfCF)) 2.*ones(1, length(diameter_CMoCF))];
boxplot(diameter_all, g, 'labels', {'CM-aCF' 'CM-fCF' 'CM only'});
ylim([0 1.1.*max(diameter_all)]);
title('Diameter of Individual Aggregates')
ylabel('micrometers')
%hold on
%scatter(g+1, CM_ind,'.')


figure;
vol_all = [vol_CMaCF vol_CMfCF vol_CMoCF];
g = [zeros(1, length(vol_CMaCF)) ones(1, length(vol_CMfCF)) 2.*ones(1, length(vol_CMoCF))];
boxplot(vol_all, g, 'labels', {'CM-aCF' 'CM-fCF' 'CM only'});
ylim([0 1.1.*max(vol_all)]);
title('Volume of Individual Aggregates')
ylabel('micrometers^3')
%hold on
%scatter(g+1, CM_ind,'.')


% -----------------------------------------------------
% Figures for CM-CF only
figure;
diameter_all = [diameter_CMaCF diameter_CMfCF diameter_CMoCF];
g = [zeros(1, (length(diameter_CMaCF) + length(diameter_CMfCF))) ones(1, length(diameter_CMoCF))];
boxplot(diameter_all, g, 'labels', {'CM-CF' 'CM only'});
ylim([0 1.1.*max(diameter_all)]);
title('Diameter of Individual Aggregates')
ylabel('micrometers')
%hold on
%scatter(g+1, CM_ind,'.')


figure;
vol_all = [vol_CMaCF vol_CMfCF vol_CMoCF];
g = [zeros(1, (length(vol_CMaCF) + length(vol_CMfCF))) ones(1, length(vol_CMoCF))];
boxplot(vol_all, g, 'labels', {'CM-CF' 'CM only'});
ylim([0 1.1.*max(vol_all)]);
title('Volume of Individual Aggregates')
ylabel('micrometers^3')

figure;
plot(g, vol_all, '*')
xlim([-0.5 1.5])
ylim([0 1.1.*max(vol_all)]);
title('Volume of Individual Aggregates: Raw Data')
ylabel('micrometers^3')

%return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
% Making schematic of the organoid

filename_unique{1} = 'a4_4color_sphere5_CF';
filename_unique{2} = 'a4_4color_sphere5_CM';


for i = 1:length(filename_unique)
    filename{i} = [filename_begin filename_unique{i} filename_mid1 filename_unique{i} filename_end];
end

%--------------------------------------------------------------------

posCM = xlsread(filename{2});%CM is usually the second file
posCF = xlsread(filename{1});%CF is the first file

mesh_pos = [posCM(:,1:3); posCF(:, 1:3)];
mesh_pos = mesh_pos - [min(mesh_pos(:,1)) min(mesh_pos(:,2)) min(mesh_pos(:,3))]; %normalizing to zeros

figure;
k = boundary(mesh_pos);
hold on
trisurf(k,mesh_pos(:,1),mesh_pos(:,2),mesh_pos(:,3),'Facecolor','blue','FaceAlpha',0.1)
axis equal
title('CM-CF organoid')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% second organoid schematic

filename_unique{1} = 'c4_4color_sphere5_CF';
filename_unique{2} = 'c4_4color_sphere5_CM';


for i = 1:length(filename_unique)
    filename{i} = [filename_begin filename_unique{i} filename_mid1 filename_unique{i} filename_end];
end

%--------------------------------------------------------------------



posCM = xlsread(filename{2});%CM is usually the second file
posCF = xlsread(filename{1});%CF is the first file

mesh_pos = [posCM(:,1:3); posCF(:, 1:3)];
mesh_pos = mesh_pos - [min(mesh_pos(:,1)) min(mesh_pos(:,2)) min(mesh_pos(:,3))]; %normalizing to zeros

figure;
k = boundary(mesh_pos);
hold on
trisurf(k,mesh_pos(:,1),mesh_pos(:,2),mesh_pos(:,3),'Facecolor','blue','FaceAlpha',0.1)
axis equal
title('CM - only organoid')




