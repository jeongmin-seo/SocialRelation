import numpy as np
import csv
import os
import glob
import progressbar
import scipy.io
from scipy.special import comb
from math import log, factorial
from itertools import chain, combinations



kLambda = 0.8
kTheta = 2.2

kNumRelations = 8
kWorkspacePath = "/home/mlpa/Workspace"
kDatasetPath = os.path.join(kWorkspacePath, "dataset/interacting_group_detection")
kResultPath = os.path.join(kWorkspacePath, "experimental_result/interaction_group_detection")
kBoxInfoDirPath = os.path.join(kWorkspacePath, "dataset/interacting_group_detection/ground_truth")

kGeoFactorPath = os.path.join(kResultPath, "geometric_factors")
kSRScorePath = os.path.join(kResultPath, "social_relational_scores")
kGroupScoreSavePath = os.path.join(kResultPath, "additional_data")


def get_group_scores(_image_path, _theta, _lambda):

    image_name = os.path.basename(image_path).replace(".jpg", "")

    # load geometric factors
    with open(os.path.join(kGeoFactorPath, image_name + ".csv")) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        geo_factors = []
        for row in reader:
            geo_factors.append([float(x) for x in row])

    # load social relational scores
    with open(os.path.join(kSRScorePath, image_name + ".csv")) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        sr_scores = []
        for row in reader:
            sr_scores.append([float(x) for x in row])

    # load ground truth
    mat_dict = {}
    mat_dict.update(scipy.io.loadmat(os.path.join(kBoxInfoDirPath, image_name + ".mat")))
    gt_groups = [set(group[0]) for group in mat_dict['Group'][0] if len(group[0]) > 1]

    # get candidate groups ([NOTICE] consists of pair ids not head ids)
    candidate_groups = get_candidate_groups(geo_factors, _theta)
    num_groups = len(candidate_groups)

    # get group scores
    print("Get scores of %d groups\n" % num_groups)
    group_scores = []
    with progressbar.ProgressBar(max_value=num_groups) as get_score_bar:
        for g_i, group in enumerate(candidate_groups):
            group_scores.append(get_group_score(group['pair_pos'], geo_factors, sr_scores, _lambda))

    group_score_info = []
    for g_idx, group in enumerate(candidate_groups):
        cur_info = {'ids': [id for id in group['head_ids']], 'score': group_scores[g_idx], 'is_gt':False}
        for gt_group in gt_groups:
            cur_info['is_gt'] = True
        group_score_info.append(cur_info)

    return group_score_info


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


def nCr(n, r):
    return factorial(n) / (factorial(r) * factorial(n-r))


def get_candidate_groups(_geo_factors, _theta):

    num_heads = int(_geo_factors[-1][1])
    head_sets = []

    # find invalid pairs
    invalid_pairs = []
    for factor in _geo_factors:
        if factor[3] > _theta:
            invalid_pairs.append((factor[0], factor[1]))

    print("Generate candidate groups with %d heads\n" % num_heads)
    with progressbar.ProgressBar(max_value=pow(2, num_heads)-1) as gen_group_bar:
        for set_i, subset in enumerate(all_subsets(list(range(0, num_heads)))):

            # # DEBUG
            # if set_i > 20:
            #     break

            if 2 > len(subset):
                continue

            # take a chance to deal with ordered list
            group_is_valid = True
            pair_pos = []
            head_ids = set()
            for id_pair in combinations(subset, 2):

                # we assumed that all pairs are tuples of (smaller id, larger id)
                if id_pair[0] > id_pair[1]:
                    id_pair = (id_pair[1], id_pair[0])

                if id_pair in invalid_pairs:
                    group_is_valid = False
                    break

                pair_pos.append(get_pair_pos(id_pair[0], id_pair[1], num_heads))
                head_ids.add(id_pair[0])
                head_ids.add(id_pair[1])

            if group_is_valid:
                head_sets.append({'pair_pos': pair_pos, 'head_ids': head_ids})
                # head_sets.append({'pair_pos': pair_pos, 'head_ids': sorted(list(head_ids))})

            gen_group_bar.update(set_i)

    return head_sets


def get_pair_pos(id1, id2, num_heads):
    return int(num_heads * id1 - (id1 * (id1 + 1)) / 2 + id2 - id1 - 1)


def get_group_score(_pair_pos_set, _geo_factors, _sr_scores, _lambda):
    # calculate group score
    geometric_score = 0
    social_relation_score = 0
    for pair_id in _pair_pos_set:
        geometric_score += -log(_geo_factors[pair_id][2])  # [id1, id2, delta, scale_ratio] in geo_factors
        social_relation_score += np.sum(_sr_scores[pair_id][2:])  # [id1, id2, sr1, sr2, ...] in sr_scores
    social_relation_score = log(social_relation_score)  # eq.7

    return _lambda * geometric_score + (1.0 - _lambda) * social_relation_score


if "__main__" == __name__:

    # read image list
    image_list = glob.glob(os.path.join(kDatasetPath, "image/*.jpg"))
    num_images = len(image_list)

    # make saving folder
    if not os.path.exists(kGroupScoreSavePath):
        os.makedirs(kGroupScoreSavePath)

    group_score_infos = []
    for i, image_path in enumerate(image_list):
        print(">> Find groups in %s ... [%03d/%03d]\n" % (os.path.basename(image_path), i, num_images))
        group_score_infos.append(get_group_scores(image_path, kTheta, kLambda))

    gt_group_score, other_group_score = [], []
    for info in group_score_infos:
        if info['is_gt']:
            gt_group_score.append(info['score'])
        else:
            other_group_score.append(info['score'])

    with open(os.path.join(kGroupScoreSavePath, 'group_scores_gt.csv')) as csv_file:
        w = csv.writer(csv_file)
        for score in gt_group_score:
            w.writerow([score])

    with open(os.path.join(kGroupScoreSavePath, 'group_scores_other.csv')) as csv_file:
        w = csv.writer(csv_file)
        for score in other_group_score:
            w.writerow([score])
