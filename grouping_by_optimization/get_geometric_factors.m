function [pairs, deltas, scale_ratios] = get_geometric_factors(box_file_path)

load(box_file_path);  % <= bbox is loaded

bbox(:,end) = [];
num_boxes = size(bbox, 1);
pairs = nchoosek(1:num_boxes, 2);
num_pairs = size(pairs, 1);

deltas = zeros(num_pairs, 1);
scale_ratios = deltas;

for i = 1:num_pairs
    box1 = bbox(pairs(i,1),:);
    box2 = bbox(pairs(i,2),:);
    l1 = 0.5 * [box1(1)+box1(3), box1(2)+box1(4)];
    l2 = 0.5 * [box2(1)+box2(3), box2(2)+box2(4)];
    s1 = box1(3);  % <= width = height, so set scale to width
    s2 = box2(3); 
    
    deltas(i) = 2 * norm(l1 - l2, 2) / (s1 + s2);
    scale_ratios(i) = max(s1, s2) / min(s1, s2);
end

end