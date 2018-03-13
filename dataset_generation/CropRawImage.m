% Crop head image from input image with box information that
% was hand labeled already - 2018.02.27- by Haanju Yoo

% //[OLD DESCRIPTION]
% //code for imgae croping by box & depth ground truth file
% //add croping to mat file for social relation network
% //created by 2017.06.28

clear all;
close all;

%====================================================================
% PREDEFINES
%====================================================================

% cropping size
kCropSize = 48;

% input image file format
kImageFormat = 'jpg';

% category information
kCategoryNames = {...
    'bus_stop', 'cafeteria', 'classroom', 'conference', ...
    'library', 'park', 'etc'};

for kTargetCategoryID = 1:length(kCategoryNames)

% input path
kDatasetBasePath = 'D:/Workspace/Dataset/DKU_group_discovery';
kImagePath = fullfile(kDatasetBasePath, 'image', kCategoryNames{kTargetCategoryID});
kBoxInfoPath = fullfile(kDatasetBasePath, 'box', kCategoryNames{kTargetCategoryID});
kPairSavePath = fullfile(kDatasetBasePath, 'head_pair_mat', kCategoryNames{kTargetCategoryID});
kDepthInfoPath = 'new/Depth_GT';


%====================================================================
% DATA FEEDING PREPERATION
%====================================================================

% load input file lists
image_file_list = dir(fullfile(kImagePath, ['*.', kImageFormat]));
box_info_file_list = dir(fullfile(kBoxInfoPath, '*.mat'));

% img_root = 'Rename2';
% GT_format='final.mat';

% image_file_list = image_file_list(3:end);
% box_info_file_list = box_info_file_list(3:end);

if ~isdir(kPairSavePath)
    mkdir(kPairSavePath);
end

% imgForYaw=[];

%% croping raw images for relation network and change to top-view for F-formation methods
%
% for M = 1:r
%
%     box_info_file_name = box_info_file_list(M).name;
%
%     %
%     img_num = strtok(box_info_file_name, 'final.mat');
%     temp_img_name = strcat(img_num, img_format);
%
%     load(box_info_file_name);
%     img = imread(temp_img_name);
%
%     [r1, c1] = size(bbox);
%
%     %croping images
%     for N1 = 1:r1
%         temp_img = imcrop(img,bbox(N1,1:end-1));
%         temp_img2 = imresize(temp_img,[70 70]);
%         img_file_name = strcat(img_num,'_', num2str(N1-1),img_format);
%         imwrite(temp_img2,img_file_name)
%     end
%
%
%
% end

%% croping image to mat file
% input for social relation network. make combination set


%====================================================================
% HEAD CROPPING LOOP
%====================================================================
for cur_box_info_file = {box_info_file_list(:).name}   
    
    % load box info
    box_file_name = cur_box_info_file{1};
    load(fullfile(kBoxInfoPath, box_file_name));  % <- bbox/ids is loaded
    num_boxes = size(bbox, 1);
    bbox(:,end) = [];  % remove id
    fprintf('File: %s (# of boxes=%d)\n', box_file_name, num_boxes);
    
    % load input image and convert to grayscale
    image_file_name = strrep(box_file_name, '.mat', '.jpg');    
    img = imread(fullfile(kImagePath, image_file_name));
    if 3 == size(img, 3)
        img = rgb2gray(img);
    end
    [img_h, img_w] = size(img);
    
    % make pairs indices
    pair_ids = nchoosek(1:num_boxes, 2);
    num_pairs = size(pair_ids, 1);
    cropped_heads = cell(2, 1);
    for i = 1:2
        cropped_heads{i} = zeros(num_pairs, kCropSize^2);
    end    
    
    % crop heads
    crop_size = [kCropSize, kCropSize];
    for pIdx = 1:num_pairs        
        for bIdx = 1:2
            cur_box = bbox(pair_ids(pIdx,bIdx),:);
            x = max(0, cur_box(1));
            y = max(0, cur_box(2));
            w = min(img_w-cur_box(1)+1, cur_box(3));
            h = min(img_h-cur_box(2)+1, cur_box(4));
            cropped_heads{bIdx}(pIdx,:) = ...
                reshape(imresize(imcrop(img,[x,y,w,h]),crop_size),1,[]);
            
%             % for visualization
%             temp = imresize(imcrop(img,[x,y,w,h]),crop_size);
%             imshow(temp);            
        end
    end
    head_1 = cropped_heads{1};
    head_2 = cropped_heads{2};
    
    % save results    
    save(fullfile(kPairSavePath, ...
        strrep(box_file_name, '.mat', '_relation.mat')), ...
        'head_1', 'head_2', 'pair_ids');
end

end


