%% code for imgae croping by box & depth ground truth file
% add croping to mat file for social relation network
% created by 2017.06.28

%% interaction realtion
% for show total result
clc; clear all; close all;

%% add path
% addpath('new')
addpath('./rawdata')
addpath('./box_results')
% addpath('new/Depth_GT')

%% get list img, GT
list_img = dir('./rawdata');
list_BoxGT = dir('./box_results');
img_format = '.jpg';

% img_root = 'Rename2';
% GT_format='final.mat';

list_img = list_img(3:end);
list_BoxGT = list_BoxGT(3:end);

[r, c] = size(list_BoxGT);
imgForYaw=[];

%% croping raw images for relation network and change to top-view for F-formation methods
%
% for M = 1:r
%
%     temp_BoxGT_name = list_BoxGT(M).name;
%
%     %
%     img_num = strtok(temp_BoxGT_name, 'final.mat');
%     temp_img_name = strcat(img_num, img_format);
%
%     load(temp_BoxGT_name);
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

crop_file_name = '_relation.mat';


for M = 109:r
    
    temp_BoxGT_name = list_BoxGT(M).name;
    
    
    img_num = strtok(temp_BoxGT_name, '.jpg.mat');
    temp_img_name = strcat(img_num, img_format);
    
    load(temp_BoxGT_name);
    img = imread(temp_img_name);
    
    [r1, c1] = size(bbox);
    
    LabelData =[];
    
    [r1, c1] = size(bbox);
    
    temp_id = unique(ids);
    [r2, c2] = size(temp_id);
    
    %make inter case
    Inter_case = nchoosek(temp_id,2);
    [r3, c3] = size(Inter_case);
    
    %make temp variable
    croped_head1 = [];
    croped_head2 = [];
    temp_croped_head1 = [];
    temp_croped_head2 = [];
    
    
    for N1 = 1:r3
        
        %get croping information(for indexing plus '1')
        temp_idx1 = bbox(Inter_case(N1,1)+1,1:end-1);
        temp_idx2 = bbox(Inter_case(N1,2)+1,1:end-1);
        
        %croping images
        temp_croped_head1 = imcrop(img,temp_idx1);
        temp_croped_head2 = imcrop(img,temp_idx2);
        
        [a11,b11,c11]=size(temp_croped_head1);
        
%         croped_head1 = [croped_head1; reshape(imresize(rgb2gray(temp_croped_head1),[48,48]),1,[])];
%         croped_head2 = [croped_head2; reshape(imresize(rgb2gray(temp_croped_head2),[48,48]),1,[])];

        
        if c11 == 1
            croped_head1 = [croped_head1; reshape(imresize(temp_croped_head1,[48,48]),1,[])];
            croped_head2 = [croped_head2; reshape(imresize(temp_croped_head2,[48,48]),1,[])];
        else
            croped_head1 = [croped_head1; reshape(imresize(rgb2gray(temp_croped_head1),[48,48]),1,[])];
            croped_head2 = [croped_head2; reshape(imresize(rgb2gray(temp_croped_head2),[48,48]),1,[])];
        end
        
        save_name = strcat(img_num,crop_file_name);
        
        save (save_name,'croped_head1','croped_head2')
        
    end
    
    
end


