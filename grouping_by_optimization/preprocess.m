% function [] = preprocess()

kBoxInfoDirPath = 'D:/Workspace/Dataset/group_detection_dataset/annotation';
kFactorSavePath = 'D:/Workspace/GitHub/SocialRelation/grouping_by_optimization/factors';

folder_list = dir(kBoxInfoDirPath);
folder_list = folder_list([folder_list.isdir]);
folder_list(1:2) = []; % remove '.' and '..'

% save form
% | id_1 | id_2 | delta | scale_ratio | sr_score_1 | ... | sr_score_8 |

for i = 1:numel(folder_list)
    
    cur_folder = fullfile(folder_list(i).folder, folder_list(i).name);
    file_list = dir(fullfile(cur_folder, '*.mat'));
    fprintf('Category: %s\n', folder_list(i).name);
    
    save_folder = fullfile(kFactorSavePath, folder_list(i).name);
    if ~isdir(save_folder)
        mkdir(save_folder);
    end
    
    for j = 1:numel(file_list)
        
        [pairs, deltas, scale_ratios] = ...
            get_geometric_factors(fullfile(cur_folder, file_list(j).name));
        
%         sr_data = csvread('', 2);
        sr_data = zeros(size(pairs, 1), 10);
        
        result_data = [pairs, deltas, scale_ratios, sr_data(:,3:end)];        
        
        % write csv with table
        T = array2table(result_data, ...
            'VariableNames', {'id_1', 'id_2', 'delta', 'scale_ratio', ...
            'sr_score_1', 'sr_score_2', 'sr_score_3', 'sr_score_4', ...
            'sr_score_5', 'sr_score_6', 'sr_score_7', 'sr_score_8'});
        save_file_name = strrep(file_list(j).name, '.mat', '.csv');
        writetable(T, fullfile(save_folder, save_file_name));
        
        fprintf('  %s is done ... [%03d/%03d]\n', file_list(j).name, j, numel(file_list));
    end
end

% end