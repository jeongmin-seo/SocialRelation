%% interaction realtion
% for show total result
clc; clear all; close all;

%% add path
% addpath('./DATA/new/Box_GT/')
% addpath('./DATA/new/img')
% addpath('./DATA/new/relation_GT')
% addpath('./DATA/new/relation_case')
%
addpath('./DATA/previous/Final_Dataset/finalGT')
addpath('./DATA/previous/Final_Dataset/Rename2')

%% get list img, GT
% list_img = dir('./DATA/new/img');
% list_GT = dir('./DATA/new/Box_GT');
%
list_img = dir('./DATA/previous/Final_Dataset/Rename2');
list_GT = dir('./DATA/previous/Final_Dataset/finalGT');

%new
GT_format='.jpg.mat';

%previous
% GT_format='final.mat';
img_format = '.jpg';

list_img = list_img(3:end);
list_GT = list_GT(3:end);

[r, c] = size(list_GT);
Result={};

%% parameter setting
alpha = 0;
beta1 = [0:0.01:1];
Scale_thre1=[1:0.1:5];

[pr2, pc2] = size(beta1);
[pr3, pc3] = size(Scale_thre1);

para_result=[];

%% repeat parameter

tic
for X = 1:pc1
    function_thre = function_thre1(X);
    for W = 1:pc2
        
        beta = beta1(W);
        for E = 1:pc3
            
            
            Scale_thre = Scale_thre1(E);
            
            %% calculate all grouping result
            
            for i = 1:r
                
                %             close all
                
                temp_GT_name = list_GT(i).name;
                img_num = strtok(temp_GT_name, 'final.mat');
                temp_img_name = strcat(img_num, img_format);
                
                %             img_num = strtok(temp_GT_name, '.jpg.mat');
                %             temp_img_name = strcat(img_num, img_format);
                %               [GT,temp_group1, temp_group2, temp_group3, temp_group4] = Grouping_vis_groubi(temp_GT_name, temp_img_name);
                [Set2, paired_energy2, GT, temp_group2, result_temp] = Grouping_vis_groubi(alpha, beta, Scale_thre, temp_GT_name, temp_img_name,img_num);
                %
                %                                drawnow
                %                                pause
                %
                %     Result{i,1} = GT;
                %     Result{i,2} = temp_group1;
                %     Result{i,3} = temp_group2;
                %     Result{i,4} = temp_group3;
                %     Result{i,5} = temp_group4;
                
                
                Result{i,1} = GT;
                Result{i,3} = temp_group2;
                
                
                
            end
            
            %% calculate accuracy, precision
            %
            % T = 2/3;
            %  [FN, TP, FP,new_GT2]= CalculAccuracy(Result,T);
            
            %% GT term change array to number
            new_GT = {};
            
            for i = 1: r
                
                temp_GT = Result{i,1};
                
                [r1,c1] = size(temp_GT);
                
                for j = 1 : c1
                    temp_temp_GT =  temp_GT{1,j};
                    
                    temp_temp_GT = cell2mat(temp_temp_GT);
                    
                    [r2, c2] = size(temp_temp_GT);
                    
                    temp_new_GT=[];
                    
                    for k = 1:c2
                        temp_new_GT = [temp_new_GT,str2num(temp_temp_GT(k))];
                    end
                    new_GT{i}{j} = temp_new_GT;
                end
            end
            
            %%
            new_GT2={};
            [r3, c3] = size(new_GT);
            for m = 1: c3
                [r4,c4]=size(new_GT{1,m});
                for n = 1:c4
                    new_GT2{m}{c4-n+1} = new_GT{m}{n};
                end
            end
            
            
            clear new_GT
            %%
            
            total_precision =[];
            total_recall=[];
            total_TP =[];
            total_FP =[];
            total_FN=[];
            
            
            for K = 1:r
                test = Result{K,3};
                test2 = new_GT2{K};
                [precision,recall,TP,FP,FN] = ff_evalgroups(test,test2,'card',0) ;
                % [precision,recall,TP,FP,FN] = ff_evalgroups(Result{:,3},new_GT2','card') ;
                
                total_precision =[total_precision;precision];
                total_recall=[total_recall;recall];
                total_TP =[total_TP;TP];
                total_FP =[total_FP;FP];
                total_FN=[total_FN,FN];
            end
            
            
            mean_precision = mean(total_precision)
            mean_recall = mean(total_recall)
            
            
            F1 = 2*((mean_precision*mean_recall)/(mean_precision+mean_recall))
            
            para_result=[para_result; alpha, beta, Scale_thre, mean_precision, mean_recall, F1];
        end
    end
end
toc
