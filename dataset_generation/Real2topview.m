%% make top-view point using Depth & pitch,yaw GT
%% created by 2017.06.28

clc; clear all; close all;

%% add data root & parameter setting
addpath('Box_GT/')
addpath('croped_img/')
addpath('Depth_GT/')
addpath('pitchyaw_GT/')

img_format = '.jpg';
pitch_format = '_pitchyaw.txt';
Depth_format = '_DepthGT.mat';
Depth_img_format = '_output.jpg';

%% getting list of box code
list_BoxGT= dir('Box_GT/');
[r,c] = size(list_BoxGT);

%% repeat number of all GT datas

for M = 179:r
    
    
    temp_BoxGT_name = list_BoxGT(M).name;
    
    % get img index
    img_num = strtok(temp_BoxGT_name, ',jpg.mat');
    
    % make image name
    temp_img_name = strcat(img_num, img_format);
    temp_img = imread(temp_img_name);
    
    % make pitch, yaw GT file name
    pitchyaw_value=load(strcat(img_num,pitch_format));

    load(temp_BoxGT_name);
    
    % make Depth GT file name
    temp_Depth = load(strcat(img_num,Depth_format));
    temp_Depth = temp_Depth.value;
    temp_Depth_img = imread(strcat(img_num,Depth_img_format));
    
    
    
    [r,c] = size(bbox);
    [t1, t2, t3] = size(temp_img);
    
    %test
    temp_Depth_img = double(temp_Depth_img);
    temp_Depth = reshape(temp_Depth,[t2,t1])';
    
    % head center
    head_center = [bbox(:,1)+bbox(:,3)/2, bbox(:,2)+bbox(:,4)/2];
    
    
    
    % make top_view point
    Depth_value2=[];
    for i = 1:r
        Depth_value2=[Depth_value2; temp_Depth(round(head_center(i,2)),round(head_center(i,1)))];
    end
    
    top_view_y2 = (t2/255)*(Depth_value2);
    top_view_point = [head_center(:,1),top_view_y2];
    
    
    % Cocatenate top-view coordinate & head orientation
    
    top_view_point = [top_view_point,pitchyaw_value];
    
    file_name = strcat(img_num,'_topview.mat');
    
    eval(['save' ' ' file_name ' ' 'top_view_point'])
   
    
%     % plot result
%     for k = 1:r
%         
%         subplot(2,2,1)
%         imshow(temp_img)
%         hold on
%         scatter(head_center(k,1),head_center(k,2),'filled')
%         
%         subplot(2,2,2)
%         imshow(uint8(temp_Depth_img))
%                 hold on
%         scatter(head_center(k,1),head_center(k,2),'filled')
%         
%         %this!
%         subplot(2,2,3)
%         scatter(top_view_point(k,1),Depth_value2(k),'filled')
%         hold on
%         
%         subplot(2,2,4)
%         scatter(top_view_point(k,1),top_view_point(k,2),'filled')
%         %     axis([min(top_view_point(:,1)) max(top_view_point(:,1)) min(top_view_point(:,2)) max(top_view_point(:,2))])
%         hold on
%         
%         
%         pause;
%         
%     end
%     
%     close all
    
end
