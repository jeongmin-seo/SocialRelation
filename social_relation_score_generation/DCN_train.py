# Relation Network Training Code
# Ver. Keras
# 2018.03.07 - modified by Haanju Yoo -

#########################################################
# LOAD PACKAGES
#########################################################
import os
import csv
import numpy as np
import scipy.io as sio
import keras.layers
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras import losses


#########################################################
# PRE-DEFINES
#########################################################
kNumRelations = 8
kWorkspacePath = "/home/mlpa/data_ssd/workspace"
kNetworkSavePath = os.path.join(kWorkspacePath, "dataset/group_detection/networks")

kRelationBasePath = '/home/user/Desktop/Workspace/relation'
kInteractionBasePath = '/home/user/Desktop/Workspace/Interaction_relation'


#########################################################
# TENSOR BOARD
#########################################################
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)


#########################################################
# DATA LOADING
#########################################################
def load_Kaggle_dataset(base_path, read_npy=True):
    if read_npy and os.path.isfile(os.path.join(base_path, "x_train.npy")):
        x_train = np.load(os.path.join(base_path, "x_train.npy"))
        y_train = np.load(os.path.join(base_path, "y_train.npy"))
        x_test = np.load(os.path.join(base_path, "x_test.npy"))
        y_test = np.load(os.path.join(base_path, "y_test.npy"))
    else:
        # if .npy files do not exist, create them with csv file
        num_lines = 35887  # hard coding for the convenient (TODO: change this to the flexible reading)
        num_train_rough = 30000
        num_test_rough = 10000
        data_file_path = os.path.join(base_path, "fer2013.csv")
        with open(data_file_path, newline='') as csvfile:
            raw_data = csv.DictReader(csvfile)
            num_train, num_test = 0, 0
            x_train = np.zeros((num_train_rough, 48, 48, 1))
            y_train = np.zeros((num_train_rough, 1))
            x_test = np.zeros((num_test_rough, 48, 48, 1))
            y_test = np.zeros((num_test_rough, 1))
            for i, row in enumerate(raw_data):
                print("processing on %d / %d" % (i, num_lines))
                emotion_idx = int(row["emotion"])
                pixels = np.array(list(map(int, row["pixels"].split()))).reshape((48, 48, 1))
                usage = row["Usage"]
                if "Training" == usage:
                    x_train[num_train, :, :, :] = pixels
                    y_train[num_train] = emotion_idx
                    num_train += 1
                else:
                    x_test[num_test, :, :, :] = pixels
                    y_test[num_test] = emotion_idx
                    num_test += 1

        # save parsing result
        np.save(os.path.join(base_path, "x_train.npy"), x_train[0:num_train, :, :, :])
        np.save(os.path.join(base_path, "y_train.npy"), y_train[0:num_train, :])
        np.save(os.path.join(base_path, "x_test.npy"), x_test[0:num_test, :, :, :])
        np.save(os.path.join(base_path, "y_test.npy"), y_test[0:num_test, :])

    # to one-hot-encode
    y_train = keras.utils.to_categorical(y_train, num_classes=7)
    y_test = keras.utils.to_categorical(y_test, num_classes=7)

    return x_train, y_train, x_test, y_test


#########################################################
# MODEL CONSTRUCTION
#########################################################
def build_deep_conv_net():
    deep_conv_net = Sequential()

    # layer 1, input size: 48 x 48 x 1
    deep_conv_net.add(Conv2D(64, 5, padding='same', activation='relu', input_shape=(48, 48, 1),
                             kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # layer 2, input size: 24 x 24 x 64
    deep_conv_net.add(Conv2D(96, 5, padding='same', activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # layer 3, input size: 12 x 12 x 96
    deep_conv_net.add(Conv2D(256, 5, padding='same', activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # layer 4, input size: 6 x 6 x 256
    deep_conv_net.add(Conv2D(256, 5, padding='same', activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # layer 5, input size: 6 x 6 x 256
    deep_conv_net.add(Flatten())
    deep_conv_net.add(Dense(2048, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.01)))
    deep_conv_net.add(BatchNormalization())
    deep_conv_net.add(Dropout(0.5))

    # output size: 2048
    return deep_conv_net


#########################################################
# MODEL SAVING
#########################################################
def save_model_to_json(save_path, model):
    with open(save_path, "w") as json_file:
        # serialize model to JSON
        json_file.write(model.to_json())
        print("model is saved at %s" % save_path)


def save_models(final_model, expression_model, relation_models):
    if not os.path.exists(kNetworkSavePath):
        os.makedirs(kNetworkSavePath)
    save_model_to_json(os.path.join(kNetworkSavePath, "final_model.json"), final_model)
    save_model_to_json(os.path.join(kNetworkSavePath, "Expression_model.json"), expression_model)
    for i, cur_model in enumerate(relation_models):
        save_model_to_json(os.path.join(kNetworkSavePath, "Relation%d_model.json" % i), cur_model)
    print("All models are saved")


#########################################################
# MODEL TRAINING
#########################################################
def train_DCN(dataset_path="dataset/Kaggle"):

    # model consctruction
    dcn = build_deep_conv_net()
    expression_net = Sequential()
    expression_net.add(dcn)
    expression_net.add(Dense(7, activation='sigmoid'))
    # model_optimizer = keras.optimizers.Adam(
    #     lr=1e-4, beta_1=0.99, beta_2=0.99, epsilon=1e-08, decay=1e-4)
    expression_net.compile(optimizer='sgd',
                           loss=losses.binary_crossentropy,
                           metrics=['accuracy'])

    # training
    x_train, y_train, x_test, y_test = load_Kaggle_dataset(os.path.join(kWorkspacePath, dataset_path))
    expression_net.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                       batch_size=120, epochs=400, verbose=1, callbacks=[tbCallBack])

    # save model
    save_model_to_json(os.path.join(dataset_path, "DCN.json"), expression_net)

if __name__ == "__main__":
    train_DCN()


#()()
#('')HAANJU.YOO
