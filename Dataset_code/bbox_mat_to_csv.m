% Save bbox read from .mat file to .csv file
% bbox: [x, y, w, h, id] -> save with [x, y, w, h] (because id can be
% inferred by row position)

class_names = {...
    'bus_stop', 'cafeteria', 'classroom', 'conference', ...
    'library', 'park', 'etc'};

kTargetClass = 6;
kReadPath = fullfile('D:/Workspace/Dataset/DKU_group_discovery/stanford_groupdataset_release/head_boxes', class_names{kTargetClass});
kWritePath = kReadPath;
% kWritePath = 'D:\Downloads\Box\stanford\cafeteria';

file_list = dir(fullfile(kReadPath, '*.mat'));
num_files = length(file_list);

for i = 1:num_files
    load(fullfile(kReadPath, file_list(i).name));
    csvwrite(...
        fullfile(kWritePath, [file_list(i).name(1:end-3), 'csv']), ...
        bbox(:,1:end-1));
    fprintf('%03d / %03d...\n', i, num_files);
end

% ()()
% ('')HAANJU.YOO
