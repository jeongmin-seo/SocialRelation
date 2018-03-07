clc; clear all; close all;

addpath('/Users/shelling/Desktop/boxing/result');
addpath('/Users/shelling/Desktop/boxing/total');

box_list = dir('/Users/shelling/Desktop/boxing/result');

box_list = box_list(3:end);

[r,c]= size(box_list);

for i = 15:20
    close all;
    
    temp_name = box_list(i).name;
    img_name = strtok(temp_name, '.mat');
    temp_img_name = strcat(img_name, '.jpg');
    
    load(box_list(i).name);
    temp_img = imread(temp_img_name);
    
    imshow(rgb2gray(temp_img))
    hold on;
    
    [r1, c1] = size(bbox);
    
    for j = 1: r1
        rectangle('position',[bbox(j,1:4)],'EdgeColor','b','LineWidth',3)
        text(bbox(j,1),bbox(j,2),num2str(bbox(j,5)),'FontSize',13,'BackgroundColor','w')
    end
    
    nameed = strcat(img_name,'_boxed.png');
    saveas(gcf,nameed)
    
end