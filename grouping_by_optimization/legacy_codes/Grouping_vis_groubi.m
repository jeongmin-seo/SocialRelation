% clc; clear all; 
% addpath('Final_Dataset/')
% addpath('Final_Dataset/finalGT/')
% addpath('Final_Dataset/Rename2/')
% 
% load('8final.mat')
% img = imread('Final_Dataset/Rename2/8.jpg');

% function [Group, group1, group2, group3, group4] = Grouping_vis_groubi(GT_name, img_name)
function [Set2, paired_energy2,Group, group2,result_groubi2] = Grouping_vis_groubi(alpha, beta, Scale_thre, GT_name, img_name, img_num)

% temp_relation_name = strcat(img_num,'_predicted_relation.mat');
% case_name = strcat(img_num,'_relation.mat' );

% load(temp_relation_name);
% load(case_name);

save_group_name = strcat(img_num,'_group');

load(GT_name);
img = imread(img_name);


energ=[];
di=[];
all_distance=[];
all_Scale_ratio =[];

Relation_case = nchoosek(bbox(:,end),2);

[r,c] = size(Relation_case);

% color map configuration
% cmap = colormap(parula(r));


%% calculate distance, ratio

for i = 1:r
    dis = eudis(...
        bbox(Relation_case(i,1)+1,1),...
        bbox(Relation_case(i,1)+1,1),...
        bbox(Relation_case(i,2)+1,1),...
        bbox(Relation_case(i,2)+1,1));
    temp_w =  [bbox(Relation_case(i,1)+1,3),bbox(Relation_case(i,2)+1,3)];
    temp_h =  [bbox(Relation_case(i,1)+1,4),bbox(Relation_case(i,2)+1,4)];

    [t1, t2] = max(temp_w);
    [t3, t4] = min(temp_w);
    ratio_w = abs(temp_w(t2)/temp_w(t4));
    
    [t5, t6] = max(temp_h);
    [t7, t8] = min(temp_h);
    
    ratio_h = abs(temp_h(t6)/temp_h(t8));
    
    temp_area = temp_w .* temp_h;
    Scale_img = sum(temp_area)/2;
    
    [area_value1, area_ind1] = max(temp_area);
    [area_value2, area_ind2] = min(temp_area);
    
    
    Scale_ratio = area_value1/area_value2;
    all_Scale_ratio = [all_Scale_ratio; Scale_ratio];
    
    all_distance = [all_distance; dis];
%      [ energ(i,:), di(i,:) ]= energy(Scale_img,ratio_w,ratio_h,dis,EXP1(i,:),EXP2(i,:),EXP3(i,:),EXP4(i,:),EXP5(i,:),EXP6(i,:),EXP7(i,:),EXP8(i,:));

% mode previou????
     [ sum_erfc_energy(i,:), sum_log_energy(i,:), log_sum_energy(i,:), erfc_sum_energy(i,:), log_dis(i,:) ]= ...
     energy(alpha, beta, Scale_thre, Scale_img, Scale_ratio ,dis,EXP1(i,:),EXP2(i,:),EXP3(i,:),EXP4(i,:),EXP5(i,:),EXP6(i,:),EXP7(i,:),EXP8(i,:));
%mode new
%         [ sum_erfc_energy(i,:), sum_log_energy(i,:), log_sum_energy(i,:), erfc_sum_energy(i,:), log_dis(i,:) ]= ...
%      energy(alpha, beta, Scale_thre, Scale_img, Scale_ratio ,dis,relation1(i,:),relation2(i,:),relation3(i,:),relation4(i,:),relation5(i,:),relation6(i,:),relation7(i,:),relation8(i,:));


end

% %% plot energy
% figure(1)
% subplot(2,3,1)
% plot(sum_erfc_energy)
% title('sum erfc energy')
% % axis([0, 6,-1,6])
% 
% subplot(2,3,2)
% plot(sum_log_energy)
% title('sum log energy')
% % axis([0 6,-1,6])
% 
% subplot(2,3,3)
% plot(log_sum_energy)
% title('log sum enegy')
% % axis([0 6,-1,6])
% 
% subplot(2,3,4)
% plot(erfc_sum_energy)
% title('erfc sum energy')
% 
% subplot(2,3,5)
% stem(all_Scale_ratio)
% title('all scale ratio')
% subplot(2,3,6)
% imshow(img)
% 
% % axis([0, 6,-1,6])



%% Sort energy
sum_erfc_energy_ind = [sum_erfc_energy, Relation_case];
sum_log_energy_ind = [sum_log_energy, Relation_case];
log_sum_energy_ind = [log_sum_energy, Relation_case];
erfc_sum_energy_ind = [erfc_sum_energy, Relation_case];

% energ_ind = [energ,Relation_case];
% energ_sorted = sort(energ);

sum_erfc_energy_sorted = sort(sum_erfc_energy);
sum_log_energy_sorted = sort(sum_log_energy);
log_sum_energy_sorted = sort(log_sum_energy);
erfc_sum_energy_sorted = sort(erfc_sum_energy);


%% optimazation using groubi
input = bbox(:,end)';
% 
% [Set1, paired_energy1, group1, result_groubi1] = calculIntergroups(input,sum_erfc_energy_ind);
% [Set2, paired_energy2, group2, result_groubi2] = calculIntergroups(input,sum_log_energy_ind);
% [Set3, paired_energy3, group3, result_groubi3] = calculIntergroups(input,log_sum_energy_ind);
% [Set4, paired_energy4, group4, result_groubi4] = calculIntergroups(input,erfc_sum_energy_ind);
% 


 [Set2, paired_energy2, group2, result_groubi2] = calculIntergroups(input,sum_log_energy_ind);


% 
%% plot paired energy
% 
% figure(2)
% subplot(2,2,1)
% stem(paired_energy1)
% title('pair sum erfc energy')
% 
% subplot(2,2,2)
% stem(paired_energy2)
% title('pair sum log energy')
% 
% subplot(2,2,3)
% stem(paired_energy3)
% title('pair log sum energy')
% 
% subplot(2,2,4)
% stem(paired_energy4)
% title('pair erfc sum energy')
% 

 %% visualization grouping
% figure(1)

[r1, c1] = size(group2);
[r2, c2] = size(bbox);
% cmap2 = colormap(parula(c1));

% subplot(1,2,1)
% imshow(img)
% for k = 1 : r2
%     rec_point = bbox(k,1:end-1);
%     rectangle('Position',rec_point,'EdgeColor','w','LineWidth',1.5);
% end
% % title('Grounmd truth haead box')
% 
% subplot(1,2,2)

%????????? ?????? ????????? ?????? ????????? ?????? 
imshow(img)
imshow(rgb2gray(img))
for i = 1: c1
    temp_group2 = group2{i};
    [r2, c2] = size(temp_group2);
    for j = 1: c2
        rec_point = bbox(temp_group2(j)+1,1:end-1);
        rectangle('Position',rec_point,'EdgeColor',cmap2(i,:),'LineWidth',5);
        group_name = ['Group', num2str(i)];
        text(rec_point(1),rec_point(2),group_name,'FontSize',10,'BackgroundColor','w')
    end
end

% 
saveas(gcf,save_group_name,'png')



%%  title('sum log energy')


% for i = 1: c1
%     temp_group1 = group1{i};
%     [r2, c2] = size(temp_group1);
%     for j = 1: c2
%         rec_point = bbox(temp_group1(j)+1,1:end-1);
%         rectangle('Position',rec_point,'EdgeColor',cmap2(i,:),'LineWidth',3);
%         group_name = ['Group', num2str(i)];
%         text(rec_point(1),rec_point(2),group_name,'FontSize',7,'BackgroundColor','w')
%     end
% end
% title('sum erfc energy')
% 
% %
% subplot(2,2,2)
% [r1, c1] = size(group2);
% imshow(img)
% cmap2 = colormap(parula(c1));
% 
% for i = 1: c1
%     temp_group2 = group2{i};
%     [r2, c2] = size(temp_group2);
%     for j = 1: c2
%         rec_point = bbox(temp_group2(j)+1,1:end-1);
%         rectangle('Position',rec_point,'EdgeColor',cmap2(i,:),'LineWidth',3);
%         group_name = ['Group', num2str(i)];
%         text(rec_point(1),rec_point(2),group_name,'FontSize',7,'BackgroundColor','w')
%     end
% end
% title('sum log energy')
% 
% %
% subplot(2,2,3)
% [r1, c1] = size(group3);
% imshow(img)
% cmap2 = colormap(parula(c1));
% 
% for i = 1: c1
%     temp_group3 = group3{i};
%     [r2, c2] = size(temp_group3);
%     for j = 1: c2
%         rec_point = bbox(temp_group3(j)+1,1:end-1);
%         rectangle('Position',rec_point,'EdgeColor',cmap2(i,:),'LineWidth',3);
%         group_name = ['Group', num2str(i)];
%         text(rec_point(1),rec_point(2),group_name,'FontSize',7,'BackgroundColor','w')
%     end
% end
% title('log sum energy')
% 
% %
% subplot(2,2,4)
% [r1, c1] = size(group4);
% imshow(img)
% cmap2 = colormap(parula(c1));
% 
% for i = 1: c1
%     temp_group4 = group4{i};
%     [r2, c2] = size(temp_group4);
%     for j = 1: c2
%         rec_point = bbox(temp_group4(j)+1,1:end-1);
%         rectangle('Position',rec_point,'EdgeColor',cmap2(i,:),'LineWidth',3);
%         group_name = ['Group', num2str(i)];
%         text(rec_point(1),rec_point(2),group_name,'FontSize',7,'BackgroundColor','w')
%     end
% end
% title('erfc sum energy')

