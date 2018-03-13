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
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.layers


#########################################################
# PRE-DEFINES
#########################################################
# modify absolute paths to be fit with your system
kNumRelations = 8
kInputSize = 48
kWorkspacePath = "/home/mlpa/data_ssd/workspace"
kProjectPath = os.path.join(kWorkspacePath, "github/SocialRelation")
kRelationBasePath = os.path.join(kWorkspacePath, "dataset/social_relation_dataset")
kNetworkSavePath = os.path.join(kProjectPath, "social_relation_score_generation/trained_models")


#########################################################
# CALLBACKS
#########################################################
class MyCbk(keras.callbacks.Callback):
    def __init__(self, model):
        super(MyCbk, self).__init__()
        self.model_to_save = model
        self.best_loss = 0

    def on_epoch_end(self, epoch, logs=None):
        cur_loss = logs.get('loss')
        if 0 == self.best_loss or self.best_loss > cur_loss:
            print('loss is improved')
            self.best_loss = cur_loss
            self.model_to_save.save('srn-improvement-epoch_%03d-loss_%.2f.h5' % (epoch, cur_loss))

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
    file_info = num_lines * [dict(file_name='', face_1_box=[], face_2_box=[], label=[])]
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
    x1 = np.zeros((num_samples, 48, 48, 1))  # left face
    x2 = np.zeros((num_samples, 48, 48, 1))  # right face
    ys = [np.zeros((num_samples, 1))] * kNumRelations
    for i, file_info in enumerate(file_infos):
        print("  sample: %04d / %04d" % (i+1, num_samples))
        input_image = Image.open(os.path.join(base_path, "img", file_info["file_name"])).convert("L")
        x1[i] = crop_head(input_image, file_info["face_1_box"])
        x2[i] = crop_head(input_image, file_info["face_2_box"])
        for j, label in enumerate(file_info["label"]):
            ys[j][i] = label
    return x1, x2, ys


def generate_npy_samples(base_path=kRelationBasePath):
    # training samples
    train_file_info = read_file_infos(os.path.join(base_path, "training.txt"))
    x1_train, x2_train, y_train = file_info_to_nparray(train_file_info)
    np.save(os.path.join(base_path, "x1_train.npy"), x1_train)
    np.save(os.path.join(base_path, "x2_train.npy"), x2_train)
    for i, cur_y in enumerate(y_train):
        np.save(os.path.join(base_path, "y%d_train.npy" % i), cur_y)

    # test samples
    test_file_info = read_file_infos(os.path.join(base_path, "testing.txt"))
    x1_test, x2_test, y_test = file_info_to_nparray(test_file_info)
    np.save(os.path.join(base_path, "x1_test.npy"), x1_test)
    np.save(os.path.join(base_path, "x2_test.npy"), x2_test)
    for i, cur_y in enumerate(y_test):
        np.save(os.path.join(base_path, "y%d_test.npy" % i), cur_y)


def load_data(base_path=kRelationBasePath):
    x1_train = np.load(os.path.join(base_path, "x1_train.npy"))
    x2_train = np.load(os.path.join(base_path, "x2_train.npy"))
    y_train = []
    for i in range(kNumRelations):
        y_train.append(np.load(os.path.join(base_path, "y%d_train.npy" % i)))

    x1_test = np.load(os.path.join(base_path, "x1_test.npy"))
    x2_test = np.load(os.path.join(base_path, "x2_test.npy"))
    y_test = []
    for i in range(kNumRelations):
        y_test.append(np.load(os.path.join(base_path, "y%d_test.npy" % i)))

    return x1_train, x2_train, y_train, x1_test, x2_test, y_test


#########################################################
# MODEL TRAINING
#########################################################
# call 'train_SRN' in main to train the model
def train_SRN(data_path=kRelationBasePath, save_path=kNetworkSavePath, ngpu=1):

    face_a = Input(shape=(48, 48, 1))
    face_b = Input(shape=(48, 48, 1))

    # deep convoluation feature
    dcn_model = load_model(os.path.join(kNetworkSavePath, "dcn.h5"))
    # dcn_model.load_weights(os.path.join(kNetworkSavePath, "weights-improvement-213-0.61.h5"))
    feature_a = dcn_model(face_a)
    feature_b = dcn_model(face_b)
    merged_vector = keras.layers.concatenate([feature_a, feature_b], axis=-1)

    # last layers
    x = Dense(256, activation='relu')(merged_vector)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layers = []
    for i in range(kNumRelations):
        output_layers.append(keras.layers.Dense(1, activation='sigmoid')(x))

    # overall model
    srn = Model(inputs=[face_a, face_b], outputs=output_layers)

    # training
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = load_data(data_path)
    callbacks_list = [tbCallBack, MyCbk(srn)]

    model_optimizer = keras.optimizers.Adam(
        lr=1e-4, beta_1=0.99, beta_2=0.99, epsilon=1e-08, decay=1e-4)

    if ngpu > 1:
        srn_parallel = multi_gpu_model(srn, gpus=ngpu)
        srn_parallel.compile(optimizer=model_optimizer, loss=keras.losses.binary_crossentropy,
                             loss_weights=[1.0] * kNumRelations,
                             metrics=['accuracy'])

        srn_parallel.fit(x=[x1_train, x2_train], y=y_train,
                         validation_data=([x1_test, x2_test], y_test),
                         batch_size=120, epochs=600, verbose=1, callbacks=callbacks_list)
    else:
        srn.compile(optimizer=model_optimizer, loss=keras.losses.binary_crossentropy,
                    loss_weights=[1.0] * kNumRelations,
                    metrics=['accuracy'])

        srn.fit(x=[x1_train, x2_train], y=y_train,
                validation_data=([x1_test, x2_test], y_test),
                batch_size=120, epochs=600, verbose=1, callbacks=callbacks_list)

    # save final result
    srn.save(os.path.join(save_path, "srn.h5"))


if __name__ == "__main__":
    # generate_npy_samples()
    train_SRN(ngpu=2)


# ()()
# ('')HAANJU.YOO
