from keras.models import Model, model_from_json, load_model
from glob import glob
import numpy as np
import os
import scipy.io as sio
import csv


#########################################################
# Pre-defines
#########################################################
kNumRelations = 8
kWorkspacePath = "/home/mlpa/data_ssd/workspace"
kSourceDirPath = os.path.join(kWorkspacePath, "github/SocialRelation/social_relation_score_generation")
kModelPath = os.path.join(kSourceDirPath, "trained_models/srn.h5")
kHeadPairBasePath = os.path.join(kWorkspacePath, "dataset/group_detection/head_pairs")
kResultSavingPath = os.path.join(kSourceDirPath, "social_relation_scores")


def load_pairs(pair_file_path, box_size=48):
    file_data = sio.loadmat(pair_file_path)
    head_1 = np.float32(file_data["head_1"].reshape((-1, box_size, box_size, 1)))
    head_2 = np.float32(file_data["head_2"].reshape((-1, box_size, box_size, 1)))
    return head_1, head_2, file_data["pair_ids"]


def get_scores(model, pair_file_path):
    head_1, head_2, head_ids = load_pairs(pair_file_path)
    sr_scores = model.predict(x=[head_1, head_2], verbose=0)
    return head_ids, sr_scores


def save_scores(head_ids, sr_scores, save_path):
    with open(save_path, "w") as outfile:
        writer = csv.writer(outfile)

        # write header
        header_str = "id_1, id2"
        for relation_idx in range(1, kNumRelations+1):
            header_str += ", score_%d" % relation_idx
        outfile.write(header_str+"\n")

        # write data
        for row in zip(head_ids, sr_scores):
            writer.writerow(row)


# def test_network(pair_path_list):
#     for file_id, cur_box_path in enumerate(pair_path_list):
#         cur_file_name = os.path.basename(cur_box_path)
#
#         # load features
#         merged_feature = get_head_pair_merged_feature(face_layer, cur_box_path)
#         num_pairs = merged_feature.shape[0] if 1 < merged_feature.ndim else 1
#         print("  Proc on '%s'...[%03d/%03d] wih %03d num_pairs"
#               % (cur_file_name, file_id + 1, num_files, num_pairs), end='')
#
#         # result container
#         relation_score_result = np.zeros((num_pairs, 2 + kNumRelations))
#
#         # relation score with concatenated feature
#         for i in range(kNumRelations):
#             relation_score_result[:, i + 2] = rel_models[i].predict(merged_feature, verbose=0).reshape(-1)
#
#         # save scores
#         save_file_path = os.path.join(result_save_dir, cur_file_name + ".csv")
#         # np.savetxt(save_file_path, relation_score_result, delimiter=",", header=result_csv_header)
#         np.savetxt(save_file_path, relation_score_result, delimiter=",")
#         print("...done!")


if __name__ == "__main__":

    # read directories
    category_names = [name for name in os.listdir(kHeadPairBasePath)
                      if os.path.isdir(os.path.join(kHeadPairBasePath, name))]
    # category_names = ["bus_stop"]

    # prepare models
    srn = load_model(kModelPath)

    # read box info
    for category_id, cur_category_name in enumerate(category_names):

        # prepare base paths
        print("Process on %s [%d/%d]..." % (cur_category_name, category_id, len(category_names)))
        head_pair_dir = os.path.join(kHeadPairBasePath, cur_category_name)
        result_save_dir = os.path.join(kResultSavingPath, cur_category_name)
        print("  pair dir: %s" % head_pair_dir)
        print("  save dir : %s" % result_save_dir)

        # make saving directory
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)

        # load box infos (of current category)
        box_file_path_list = glob(os.path.join(head_pair_dir, "*_relation.mat"))
        num_files = len(box_file_path_list)

        # ^ (head_id_1, head_id_2, relation_score_1, relation_score_2, ... )
        for file_id, cur_box_path in enumerate(box_file_path_list):

            pair_ids, scores = get_scores(srn, cur_box_path)

            # save scores
            cur_file_name = os.path.basename(cur_box_path).replace("_relation.mat", "")
            save_file_path = os.path.join(result_save_dir, cur_file_name + ".csv")

            save_scores(pair_ids, scores, save_file_path)

            print("  Proc on '%s' is done [%03d/%03d]" % (cur_file_name, file_id+1, num_files))

# ()()
# ('')HAANJU.YOO
