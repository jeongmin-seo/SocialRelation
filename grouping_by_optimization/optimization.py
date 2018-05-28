import numpy as np
import csv
import os
import glob
import progressbar
import time
from gurobipy import *
from math import log, factorial
from itertools import chain, combinations
import threading


kLambda = 0.8
kTheta = 2.2

kNumRelations = 8
kWorkspacePath = "/home/mlpa/Workspace"
kDatasetPath = os.path.join(kWorkspacePath, "dataset/interacting_group_detection")
kResultPath = os.path.join(kWorkspacePath, "experimental_result/interaction_group_detection")

kGeoFactorPath = os.path.join(kResultPath, "geometric_factors")
kSRScorePath = os.path.join(kResultPath, "social_relational_scores")

kGroupSavePath = os.path.join(kResultPath, "grouping_result")


incompatible_group_ids = []
thread_lock = threading.Lock()


def thread_task(_groups, _target_id, _comparing_start, _comparing_end):
    # print("Thread for %d group is started\n" % _target_id)
    incompatible_groups = []
    for id2 in range(_comparing_start, _comparing_end):
        if not _groups[_target_id]['head_ids'].isdisjoint(_groups[id2]['head_ids']):
            incompatible_groups.append((_target_id, id2))

    global incompatible_group_ids, thread_lock
    thread_lock.acquire()
    incompatible_group_ids += incompatible_groups
    thread_lock.release()
    # print("Thread for %d group is done\n" % _target_id)


def construct_and_solve_optimization_problem(_image_path, _theta, _lambda):

    time_const_opt_start = time.time()

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

    # get candidate groups ([NOTICE] consists of pair ids not head ids)
    candidata_groups = get_candidate_groups(geo_factors, _theta)
    num_groups = len(candidata_groups)

    # get group scores
    print("Get scores of %d groups\n" % num_groups)
    group_scores = []
    effective_groups = []
    with progressbar.ProgressBar(max_value=num_groups) as get_score_bar:
        for g_i, group in enumerate(candidata_groups):
            cur_score = get_group_score(group['pair_pos'], geo_factors, sr_scores, _lambda)
            if 0.0 <= cur_score:
                effective_groups.append(group)
                group_scores.append(cur_score)
            get_score_bar.update(g_i)

    num_groups = len(group_scores)
    print("Total %d effective groups\n" % num_groups)
    grouping_result = []

    if 0 == num_groups:
        save_grouping_result(grouping_result, os.path.join(kGroupSavePath, image_name + '.txt'))

        # logging
        with open(os.path.join(kGroupSavePath, image_name + "_log.txt"), "w") as text_file:
            text_file.write("num faces: %d\n" % int(geo_factors[-1][1]))
            text_file.write("num candidate groups: %d\n" % num_groups)
            text_file.write("num constraints: %d\n" % 0)
            text_file.write("model construction time: %f\n" % 0)
            text_file.write("solving time: %f\n" % 0)

        return grouping_result

    # grouping with optimization
    try:

        # Create a new model
        grb_model = Model("bip")

        # Create variables
        grb_vars = [grb_model.addVar(vtype=GRB.BINARY, name="x_%d" % g_i) for g_i in range(num_groups)]

        # Set objective
        obj = LinExpr()
        for g_i in range(num_groups):
            obj += group_scores[g_i] * grb_vars[g_i]
        grb_model.setObjective(obj, GRB.MAXIMIZE)

        # Add constraints
        global incompatible_group_ids
        threads = []
        incompatible_group_ids = []
        for g_idx_1 in range(num_groups-1):
            t = threading.Thread(target=thread_task, args=(effective_groups, g_idx_1, g_idx_1+1, num_groups))
            t.daemon = True
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        num_consts = len(incompatible_group_ids)
        print("Add %d constratins to the model\n" % num_consts)
        with progressbar.ProgressBar(max_value=num_consts) as const_bar:
            for ip_i, pair in enumerate(incompatible_group_ids):
                grb_model.addConstr(grb_vars[pair[0]] + grb_vars[pair[1]] <= 1, "c_%d" % ip_i)
                const_bar.update(ip_i)

        time_const_opt_end = time.time()

        print("Start optimization...\n")
        grb_model.optimize()

        time_solve_end = time.time()

        for v_i, v in enumerate(grb_model.getVars()):
            if 0.5 < v.x:
                grouping_result.append(effective_groups[v_i])

        print('Obj:', grb_model.objVal)

        grouping_result = result_packaging(grouping_result)

        save_grouping_result(grouping_result, os.path.join(kGroupSavePath, image_name + '.txt'))

        # logging
        with open(os.path.join(kGroupSavePath, image_name + "_log.txt"), "w") as text_file:
            text_file.write("num faces: %d\n" % int(geo_factors[-1][1]))
            text_file.write("num candidate groups: %d\n" % num_groups)
            text_file.write("num constraints: %d\n" % num_consts)
            text_file.write("model construction time: %f\n" % (time_const_opt_end - time_const_opt_start))
            text_file.write("solving time: %f\n" % (time_solve_end - time_const_opt_end))


    except GurobiError:
        print('Error reported')

    return grouping_result


def result_packaging(_result_groups, _num_heads):
    new_result = [group for group in _result_groups]
    for head_idx in range(_num_heads):
        is_found = False
        for group in _result_groups:
            if head_idx in group:
                is_found = True
                break
        if not is_found:
            new_result.append(set([head_idx]))
    return new_result


def save_grouping_result(_groups, _save_path):
    with open(_save_path, "w") as text_file:
        for group in _groups:
            for head in group['head_ids']:
                text_file.write("%d " % head)
            text_file.write("\n")


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


def nCr(n, r):
    return factorial(n) / (factorial(r) * factorial(n-r))


def disjoint_sorted_list(list1, list2):
    pos1, pos2 = 0, 0
    while pos1 < len(list1) and pos2 < len(list2):
        if list1[pos1] < list2[pos2]:
            pos1 += 1
        elif list1[pos1] > list2[pos2]:
            pos2 += 1
        else:
            return False
    return True


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
    if not os.path.exists(kGroupSavePath):
        os.makedirs(kGroupSavePath)

    for i, image_path in enumerate(image_list):
        print(">> Find groups in %s ... [%03d/%03d]\n" % (os.path.basename(image_path), i, num_images))
        groups = construct_and_solve_optimization_problem(image_path, kTheta, kLambda)
