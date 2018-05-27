import numpy as np
import glob
import os
import scipy.io
import progressbar
import csv
from numpy import linalg as LA
from scipy.special import comb

kBoxInfoDirPath = '/home/mlpa/data_ssd/workspace/dataset/interacting_group_detection/ground_truth'
kFactorSavePath = 'factors'


def get_geometric_factors(_bbox):
    num_boxes = _bbox.shape[0]
    num_pairs = int(comb(num_boxes, 2))
    pair_pos = 0
    geo_factors_of_pairs = np.zeros((num_pairs, 4))  # id1, id2, delta, scale_ratio

    for id_1 in range(0, num_boxes-1):

        # box 1 geo info
        box_1 = _bbox[id_1]
        location_1 = 0.5 * np.array([box_1[0] + box_1[2], box_1[1] + box_1[3]])
        box_size_1 = box_1[2]

        for id_2 in range(id_1+1, num_boxes):

            # box 2 geo info
            box_2 = _bbox[id_2][0:-1]
            location_2 = 0.5 * np.array([box_2[0] + box_2[2], box_2[1] + box_2[3]])
            box_size_2 = box_2[2]

            # calculate geo factors
            delta = 2 * LA.norm(location_1 - location_2, 2) / (box_size_1 + box_size_2)
            scale_ratio = max(box_size_1, box_size_2) / min(box_size_1, box_size_2)

            # save geo factors
            geo_factors_of_pairs[pair_pos][0] = id_1
            geo_factors_of_pairs[pair_pos][1] = id_2
            geo_factors_of_pairs[pair_pos][2] = delta
            geo_factors_of_pairs[pair_pos][3] = scale_ratio

            # relocate position indicator
            pair_pos += 1

    return geo_factors_of_pairs


if "__main__" == __name__:

    # save form
    # | id_1 | id_2 | delta | scale_ratio | sr_score_1 | ... | sr_score_8 |

    gt_file_path_list = glob.glob(os.path.join(kBoxInfoDirPath, "*.mat"))

    if 0 < len(gt_file_path_list) and not os.path.exists(kFactorSavePath):
        os.makedirs(kFactorSavePath)
        print("Saving path: %s\n" % os.path.abspath(kFactorSavePath))

    with progressbar.ProgressBar(max_value=len(gt_file_path_list)) as bar:
        for i, gt_file_path in enumerate(gt_file_path_list):
            mat = scipy.io.loadmat(gt_file_path)
            bbox = np.array(mat['bbox'])

            geo_factors = get_geometric_factors(bbox)
            sr_scores = np.zeros((geo_factors.shape[0], 8))
            data = np.concatenate((geo_factors, sr_scores), axis=1)

            with open(os.path.join(kFactorSavePath, os.path.basename(gt_file_path).replace(".mat", ".csv")), 'w', newline='\n') as f:
                w = csv.writer(f)
                w.writerow(['id_1', 'id_2', 'delta', 'scale_ratio',
                          'sr_score_1', 'sr_score_2', 'sr_score_3', 'sr_score_4',
                          'sr_score_5', 'sr_score_6', 'sr_score_7', 'sr_score_8'])
                w.writerows(data)

            bar.update(i)

# ()()
# ('') HAANJU.YOO
