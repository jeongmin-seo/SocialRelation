from keras.models import Model, model_from_json
from keras.layers import Merge, Input
from glob import glob
from PIL import Image
import numpy as np
import os
import sys
import scipy.io as sio


#########################################################
# Pre-defines
#########################################################
kNumRelations = 8
kWorkspacePath = "/home/mlpa/data_ssd/workspace"
kModelPath = os.path.join(kWorkspacePath, "dataset/group_detection/networks")
kHeadPairBasePath = os.path.join(kWorkspacePath, "dataset/group_detection/head_pairs")
kResultSavingPath = os.path.join(kWorkspacePath, "dataset/group_detection/relation_scores/stanford")
# kCategoryNames = ["bus_stop", "cafeteria", "classroom", "conference", "library", "park"]


#########################################################
# PART. social relation predicting for grouping
#########################################################
def load_models(base_path=kModelPath):
    # load final model
    # final_model_json_file = open(os.path.join(base_path, 'final_model_json.json'), 'r')
    # final_model_json = final_model_json_file.read()
    # final_model_json_file.close()
    # final_model = model_from_json(final_model_json)
    # final_model.load_weights(os.path.join(base_path, 'final_model.h5'))

    # expression model
    print("Loading expression model...", end='')
    expression_model_json_file = open(os.path.join(base_path, "Expression_model.json"))
    expression_model_json = expression_model_json_file.read()
    expression_model_json_file.close()
    expression_model = model_from_json(expression_model_json)
    expression_model.load_weights(os.path.join(base_path, "Expression_model.h5"))
    print("done!")

    relation_models = []
    print("Loading relation model...", end='')
    for model_id in range(1, kNumRelations+1):
        print("%d..." % model_id, end='')
        model_json_file = open(os.path.join(base_path, "Relation%d_model.json" % model_id))
        model_json = model_json_file.read()
        model_json_file.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(os.path.join(base_path, "Relation%d_model.h5" % model_id))
        relation_models.append(loaded_model)
    print("done!")

    return expression_model, relation_models


def get_head_pair_merged_feature(feature_model, file_path, box_size=48):
    file_data = sio.loadmat(file_path)
    cropped_head_1 = file_data["croped_head1"]
    cropped_head_2 = file_data["croped_head2"]
    feature_1 = feature_model.predict(cropped_head_1.reshape((-1, box_size, box_size, 1)))
    feature_2 = feature_model.predict(cropped_head_2.reshape((-1, box_size, box_size, 1)))
    # return merged feature
    return np.concatenate((feature_1, feature_2), axis=1)

def test_network(pair_path_list):
    for file_id, cur_box_path in enumerate(pair_path_list):
        cur_file_name = os.path.basename(cur_box_path)

        # load features
        merged_feature = get_head_pair_merged_feature(face_layer, cur_box_path)
        num_pairs = merged_feature.shape[0] if 1 < merged_feature.ndim else 1
        print("  Proc on '%s'...[%03d/%03d] wih %03d num_pairs"
              % (cur_file_name, file_id + 1, num_files, num_pairs), end='')

        # result container
        relation_score_result = np.zeros((num_pairs, 2 + kNumRelations))

        # relation score with concatenated feature
        for i in range(kNumRelations):
            relation_score_result[:, i + 2] = rel_models[i].predict(merged_feature, verbose=0).reshape(-1)

        # save scores
        save_file_path = os.path.join(result_save_dir, cur_file_name + ".csv")
        # np.savetxt(save_file_path, relation_score_result, delimiter=",", header=result_csv_header)
        np.savetxt(save_file_path, relation_score_result, delimiter=",")
        print("...done!")

if __name__ == "__main__":

    # read directories
    # category_names = [name for name in os.listdir(kHeadInfoBasePath)
    #                   if os.path.isdir(os.path.join(kHeadInfoBasePath, name))]
    category_names = ["bus_stop"]

    # prepare models
    expr_model, rel_models = load_models()
    face_layer = Model(inputs=expr_model.input, outputs=expr_model.get_layer(index=22).output)

    # # output related
    # result_csv_header = "head 1 id,head 2 id,"
    # for relation_id in range(kNumRelations):
    #     result_csv_header += "score %d"
    #     if relation_id not is (kNumRelations - 1):
    #         result_csv_header += ","

    # read box info
    for category_id, cur_category_name in enumerate(category_names):

        # prepare base paths
        print("Process on %s [%d/%d]..." % (cur_category_name, category_id, len(category_names)))
        box_info_dir = os.path.join(kHeadInfoBasePath, cur_category_name)
        head_pair_dir = os.path.join(kHeadPairBasePath, cur_category_name)
        result_save_dir = os.path.join(kResultSavingPath, cur_category_name)
        print("  box dir  : %s" % box_info_dir)
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
            cur_file_name = os.path.basename(cur_box_path)

            # load features
            merged_feature = get_head_pair_merged_feature(face_layer, cur_box_path)
            num_pairs = merged_feature.shape[0] if 1 < merged_feature.ndim else 1
            print("  Proc on '%s'...[%03d/%03d] wih %03d num_pairs"
                  % (cur_file_name, file_id + 1, num_files, num_pairs), end='')

            # result container
            relation_score_result = np.zeros((num_pairs, 2 + kNumRelations))

            # relation score with concatenated feature
            for i in range(kNumRelations):
                relation_score_result[:, i+2] = rel_models[i].predict(merged_feature, verbose=0).reshape(-1)

            # save scores
            save_file_path = os.path.join(result_save_dir, cur_file_name + ".csv")
            # np.savetxt(save_file_path, relation_score_result, delimiter=",", header=result_csv_header)
            np.savetxt(save_file_path, relation_score_result, delimiter=",")
            print("...done!")


#()()
#('')HAANJU.YOO
