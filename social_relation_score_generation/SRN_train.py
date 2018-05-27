# Social Relation Network Training Code
# Ver. Keras
# 2018.03.13 - by Haanju Yoo -
#
# ==<HOW TO USE>==
#  After modifying absolute paths in 'PRE-DEFINES' section,
# call 'generate_npy_samples' to make .npy data and call
# 'train_SRN' to train the Social Relation Model. Then, the
# trained model will be saved at 'kNetworkSavePath'. Note
# that load 'srn.h5' can reconstruct the model architecture,
# but load 'srn-weights.....h5' only cannot reconstruct the
# model architecture. So, you need to load 'srn.h5' first,
# and update the latest 'srn-weights....h5' because that
# weights score the best on the test set.
#
# ()()
# ('') HAANJU.YOO

#########################################################
# LOAD PACKAGES
#########################################################
import os
import numpy as np
from PIL import Image
from keras.models import load_model, Model
from keras.layers import Dense, Input, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.layers
import progressbar


#########################################################
# PRE-DEFINES
#########################################################
# modify absolute paths to be fit with your system
kNumRelations = 8
kInputSize = 48
kWorkspacePath = "/home/mlpa/data_ssd/workspace"
kProjectPath = os.path.join(kWorkspacePath, "github/SocialRelation")
kRelationBasePath = os.path.join(kWorkspacePath, "dataset/social_relation_dataset")
kNetworkSavePath = os.path.join(kWorkspacePath, "experimental_result/interaction_group_detection/trained_networks")

kSampleSetNames = ["training", "testing"]

kMinimumEpochForSaveNetwork = 10


#########################################################
# CALLBACKS
#########################################################
class MyCbk(keras.callbacks.Callback):
    def __init__(self, model):
        super(MyCbk, self).__init__()
        self.model_to_save = model
        self.best_acc = 0
        pass

    def on_epoch_end(self, epoch, logs=None):

        if kMinimumEpochForSaveNetwork > epoch:
            return

        cur_acc = logs.get('val_acc')
        if 0 == self.best_loss or self.best_loss > cur_acc:
            print('validation accuracy is improved')
            self.best_acc = cur_acc
            self.model_to_save.save('srn-improvement-epoch_%03d-acc_%.2f.h5' % (epoch, cur_acc))
        pass


tbCallBack = TensorBoard(log_dir=os.path.join(kProjectPath, 'logs'),
                         histogram_freq=0, write_graph=True, write_images=True)


#########################################################
# DATA HANDLING
#########################################################
# call 'generate_npy_samples' to prepare .npy data and
# call 'load_data' to feed the data to a model
def read_file_infos(file_path):
    fp = open(file_path, 'r')
    lines = fp.readlines()
    num_lines = len(lines)
    file_info = [dict(file_name='', face_1_box=[], face_2_box=[], label=[]) for _ in range(num_lines)]

    for i, line in enumerate(lines):
        print("%s: %04d/%04d" % (file_path, i+1, num_lines))
        line_items = line.split()
        file_info[i]["file_name"] = line_items[0]
        file_info[i]["face_1_box"] = list(map(int, line_items[1:5]))
        file_info[i]["face_2_box"] = list(map(int, line_items[5:9]))
        file_info[i]["label"] = list(map(int, line_items[9:]))

    return file_info


def crop_head(image, bbox):
    x1 = max([0, bbox[0]])
    x2 = min([image.size[0], bbox[0] + bbox[2] - 1])
    y1 = max([0, bbox[1]])
    y2 = min([image.size[1], bbox[1] + bbox[3] - 1])
    crop_area = (x1, y1, x2, y2)
    # crop head patch from input image
    input_patch = np.array(image.crop(crop_area).resize((kInputSize, kInputSize), Image.NEAREST), dtype="uint8")
    input_patch = input_patch.reshape((kInputSize, kInputSize, 1))
    return input_patch


def file_info_to_nparray(file_infos, base_path=kRelationBasePath):
    num_samples = len(file_infos)
    x1s = np.zeros((num_samples, 48, 48, 1))  # left face
    x2s = np.zeros((num_samples, 48, 48, 1))  # right face
    ys = np.zeros((num_samples, kNumRelations))
    spatial_cues = np.zeros((num_samples, 11))

    print("Generate %d samples with file infos\n" % num_samples)
    with progressbar.ProgressBar(max_value=num_samples) as bar:
        for i, file_info in enumerate(file_infos):
            input_image = Image.open(os.path.join(base_path, "img", file_info["file_name"])).convert("L")
            x1s[i] = crop_head(input_image, file_info["face_1_box"])
            x2s[i] = crop_head(input_image, file_info["face_2_box"])
            ys[i] = np.array(file_info["label"])
            # for j, label in enumerate(file_info["label"]):
            #     ys[j][i] = label

            # spatial cues
            image_width, image_height = input_image.size

            rel_x1, rel_y1 = file_info["face_1_box"][0] / image_width, file_info["face_1_box"][1] / image_height
            rel_w1, rel_h1 = file_info["face_1_box"][2] / image_width, file_info["face_1_box"][3] / image_height

            rel_x2, rel_y2 = file_info["face_2_box"][0] / image_width, file_info["face_2_box"][1] / image_height
            rel_w2, rel_h2 = file_info["face_2_box"][2] / image_width, file_info["face_2_box"][3] / image_height

            rel_x_diff = (rel_x1 - rel_x2) / rel_w1
            rel_y_diff = (rel_y1 - rel_y2) / rel_h1
            rel_size_diff = rel_w1 / rel_w2

            spatial_cues[i] = np.array([rel_x1, rel_y1, rel_w1, rel_h1,
                                        rel_x2, rel_y2, rel_w2, rel_h2,
                                        rel_x_diff, rel_y_diff, rel_size_diff])
            bar.update(i)

    return x1s, x2s, ys, spatial_cues


def generate_npy_samples(base_path=kRelationBasePath):

    for name in kSampleSetNames:
        file_info_path = read_file_infos(os.path.join(base_path, name + ".txt"))
        x1s, x2s, ys, spatial_cues = file_info_to_nparray(file_info_path)
        np.save(os.path.join(base_path, "x1s_%s.npy" % name), x1s)
        np.save(os.path.join(base_path, "x2s_%s.npy" % name), x2s)
        np.save(os.path.join(base_path, "ys_%s.npy" % name), ys)
        np.save(os.path.join(base_path, "spatial_cues_%s.npy" % name), spatial_cues)
    pass


def load_data(base_path=kRelationBasePath, set_name=kSampleSetNames[0]):
    x1s = np.load(os.path.join(base_path, "x1s_%s.npy" % set_name))
    x2s = np.load(os.path.join(base_path, "x2s_%s.npy" % set_name))
    ys = np.load(os.path.join(base_path, "ys_%s.npy" % set_name))
    spatial_cues = np.load(os.path.join(base_path, "spatial_cues_%s.npy" % set_name))

    return x1s, x2s, ys, spatial_cues


#########################################################
# MODEL TRAINING
#########################################################
# call 'train_SRN' in main to train the model
def train_SRN(data_path=kRelationBasePath, save_path=kNetworkSavePath, ngpu=1):

    face_a = Input(shape=(48, 48, 1))
    face_b = Input(shape=(48, 48, 1))
    spatial_cue = Input(shape=(11,))

    # deep convoluation feature
    dcn_model = load_model(os.path.join(kNetworkSavePath, "dcn.h5"))
    # dcn_model.load_weights(os.path.join(kNetworkSavePath, "weights-improvement-213-0.61.h5"))
    feature_a = dcn_model(face_a)
    feature_b = dcn_model(face_b)
    merged_vector = keras.layers.concatenate([feature_a, feature_b, spatial_cue], axis=-1)

    # last layers
    x = Dense(256, kernel_initializer='glorot_normal', kernel_regularizer=keras.regularizers.l2(0.01))(merged_vector)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    relations = Dense(kNumRelations, activation='sigmoid', kernel_initializer='glorot_normal')(x)

    # overall model
    srn = Model(inputs=[face_a, face_b, spatial_cue], outputs=relations)
    callbacks_list = [tbCallBack, MyCbk(srn)]

    # data preperation
    x1_train, x2_train, y_train, sc_train = load_data(data_path, kSampleSetNames[0])
    x1_test, x2_test, y_test, sc_test = load_data(data_path, kSampleSetNames[1])

    # training
    model_optimizer = keras.optimizers.Adam(
        lr=1e-4, beta_1=0.99, beta_2=0.99, epsilon=1e-08, decay=1e-4)

    if ngpu > 1:
        srn_parallel = multi_gpu_model(srn, gpus=ngpu)
        srn_parallel.compile(optimizer=model_optimizer, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
        srn_parallel.fit(x=[x1_train, x2_train, sc_train], y=y_train,
                         validation_data=([x1_test, x2_test, sc_test], y_test),
                         batch_size=120, epochs=600, verbose=1, callbacks=callbacks_list)
    else:
        srn.compile(optimizer=model_optimizer, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
        srn.fit(x=[x1_train, x2_train, sc_train], y=y_train,
                validation_data=([x1_test, x2_test, sc_test], y_test),
                batch_size=120, epochs=600, verbose=1, callbacks=callbacks_list)

    # save final result
    srn.save(os.path.join(save_path, "srn.h5"))


def test_SRN(model_path=os.path.join(kRelationBasePath, "srn.h5"), data_path=kNetworkSavePath):
    srn = load_model(model_path)
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = load_data(data_path)
    y_train_pred = srn.predict(x=[x1_train, x2_train])
    y_test_pred = srn.predict(x=[x1_test, x2_test])
    print("test done!")


if __name__ == "__main__":
    # generate_npy_samples()
    train_SRN(ngpu=1)
    # test_SRN(os.path.join(kRelationBasePath, "srn.h5"))

# ()()
# ('')HAANJU.YOO
