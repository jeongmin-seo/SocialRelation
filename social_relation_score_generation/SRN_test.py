from keras.models import Model, model_from_json, load_model
from glob import glob
import numpy as np
import os
import csv
import progressbar


#########################################################
# Pre-defines
#########################################################
kNumRelations = 8
kWorkspacePath = "/home/mlpa/Workspace"
kDatasetPath = os.path.join(kWorkspacePath, "dataset/interacting_group_detection")
kPairSamplePath = os.path.join(kDatasetPath, "pair_samples")
kResultPath = os.path.join(kWorkspacePath, "experimental_result/interaction_group_detection")
kSavePath = os.path.join(kResultPath, "social_relational_scores")
kModelPath = os.path.join(kResultPath, "trained_networks/srn.h5")


if __name__ == "__main__":

    if not os.path.exists(kPairSamplePath):
        print("No such a directory as %s\n" % kPairSamplePath)
        exit()

    # load model
    if not os.path.exists(kModelPath):
        print("No such a model file as %s\n" % kModelPath)
        exit()
    srn = load_model(kModelPath)
    print("Model is loaded\n")

    if not os.path.exists(kSavePath):
        os.makedirs(kSavePath)
        print("Saving folder is created at %s\n" % kSavePath)

    # read image list
    image_list = glob(os.path.join(kDatasetPath, "image/*.jpg"))
    num_images = len(image_list)

    with progressbar.ProgressBar(max_value=num_images) as bar:
        for i, image_path in enumerate(image_list):
            image_name = os.path.basename(image_path).replace(".jpg", "")

            # check sample data existence
            if not os.path.exists(os.path.join(kPairSamplePath, "%s_x1s.npy" % image_name)):
                print("[WARNING] There is no sample for image %s\n" % image_name)
                continue

            # load sample datas
            x1s = np.load(os.path.join(kPairSamplePath, "%s_x1s.npy" % image_name))
            x2s = np.load(os.path.join(kPairSamplePath, "%s_x2s.npy" % image_name))
            spatial_cues = np.load(os.path.join(kPairSamplePath, "%s_scs.npy" % image_name))
            ids = np.load(os.path.join(kPairSamplePath, "%s_ids.npy" % image_name))

            # get scores
            sr_scores = srn.predict(x=[x1s, x2s, spatial_cues], verbose=0)

            # save scores
            result_data = np.concatenate((ids, sr_scores), axis=1)
            file_save_path = os.path.join(kSavePath, image_name + ".csv")
            with open(file_save_path, "w") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(['id_1', 'id_2',
                                 'sr_score_1', 'sr_score_2', 'sr_score_3', 'sr_score_4',
                                 'sr_score_5', 'sr_score_6', 'sr_score_7', 'sr_score_8'])
                writer.writerows(result_data)

            bar.update(i)

# ()()
# ('')HAANJU.YOO
