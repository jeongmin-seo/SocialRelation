from keras.models import Model, model_from_json
from glob import glob
from PIL import Image
import numpy as np
import os
import sys


#########################################################
# Pre-defines
#########################################################
kNumRelations = 8
kWorkspacePath = "/home/mlpa/data_ssd/workspace"
kModelPath = os.path.join(kWorkspacePath, "dataset/group_detection/networks")
kHeadInfoBasePath = os.path.join(kWorkspacePath, "dataset/group_detection/head_boxes/stanford")
kInputImageBasePath = os.path.join(kWorkspacePath, "dataset/group_detection/images/stanford")
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


def load_box_infos(path):
    box_file_path_list = glob(os.path.join(path, "*.csv"))
    box_infos = []
    for cur_path in box_file_path_list:
        bbox = np.loadtxt(open(cur_path, "rb"), delimiter=",", skiprows=0)
        image_name = os.path.splitext(os.path.basename(cur_path))[0]
        bbox_dict = dict(file_name=image_name, bbox=bbox)
        box_infos.append(bbox_dict)
    return box_infos


def get_head_feature(feature_model, image, bbox, box_size=48):
    x1 = max([0, bbox[0]])
    x2 = min([image.size[0], bbox[0]+bbox[2]-1])
    y1 = max([0, bbox[1]])
    y2 = min([image.size[1], bbox[1]+bbox[3]-1])
    crop_area = (x1, y1, x2, y2)
    # crop head patch from input image
    input_patch = np.array(image.crop(crop_area).resize((box_size, box_size), Image.NEAREST), dtype="uint8")
    input_patch = input_patch.reshape((1, box_size, box_size, 1))
    # predict with head patch
    return feature_model.predict(input_patch)


if __name__ == "__main__":

    # read directories
    category_names = [name for name in os.listdir(kHeadInfoBasePath)
                      if os.path.isdir(os.path.join(kHeadInfoBasePath, name))]

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
        input_image_dir = os.path.join(kInputImageBasePath, cur_category_name)
        result_save_dir = os.path.join(kResultSavingPath, cur_category_name)
        print("  box dir  : %s" % box_info_dir)
        print("  image dir: %s" % input_image_dir)
        print("  save dir : %s" % result_save_dir)

        # make saving directory
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)

        # load box infos (of current category)
        box_infos = load_box_infos(box_info_dir)
        num_files = len(box_infos)

        # ^ (head_id_1, head_id_2, relation_score_1, relation_score_2, ... )
        for file_id, cur_box_info in enumerate(box_infos):
            # load input image for head patch cropping
            input_image_file_path = os.path.join(input_image_dir, cur_box_info["file_name"] + ".jpg")
            input_image = Image.open(input_image_file_path).convert("L")

            num_boxes = cur_box_info["bbox"].shape[0] if 1 < cur_box_info["bbox"].ndim else 1
            num_combinations = int(num_boxes * (num_boxes - 1) / 2)  # nC2 = n * (n-1) / 2!
            print("  Proc on '%s'...[%03d/%03d] wih %03d combinations"
                  % (cur_box_info["file_name"], file_id + 1, num_files, num_combinations))
            if num_boxes < 2:
                continue

            # result container

            relation_score_result = np.zeros((num_combinations, 2 + kNumRelations))
            combination_id = 0

            # generate scores of each combination
            for idx1 in range(num_boxes-1):
                # CNN feature extraction
                head_feature_1 = get_head_feature(face_layer, input_image, cur_box_info["bbox"][idx1,:])
                for idx2 in range(idx1+1, num_boxes):
                    # CNN feature extraction
                    head_feature_2 = get_head_feature(face_layer, input_image, cur_box_info["bbox"][idx2,:])

                    # head ids
                    relation_score_result[combination_id][0] = idx1
                    relation_score_result[combination_id][1] = idx2

                    # relation score with concatenated feature
                    merged_feature = np.concatenate((head_feature_1, head_feature_2), axis=1)
                    for i in range(kNumRelations):
                        relation_score_result[combination_id][i+2] = rel_models[i].predict(merged_feature, verbose=0)
                    # container position increment
                    combination_id = combination_id + 1
                    print("  Proc on '%s'...[%03d/%03d]...[%03d/%03d]" % (cur_box_info["file_name"], file_id+1, num_files, combination_id, num_combinations))

            # save scores
            save_file_path = os.path.join(result_save_dir, cur_box_info["file_name"] + ".csv")
            # np.savetxt(save_file_path, relation_score_result, delimiter=",", header=result_csv_header)
            np.savetxt(save_file_path, relation_score_result, delimiter=",")
            print("  Proc on '%s'...[%03d/%03d]...done!" % (cur_box_info["file_name"], file_id+1, num_files))


#()()
#('')HAANJU.YOO
