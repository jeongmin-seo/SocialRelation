# Relation Network Training Code
# Ver. Keras
# 2018.03.07 - modified by Haanju Yoo -

#########################################################
# LOAD PACKAGES
#########################################################
import os
import numpy as np
import scipy.io as sio
import keras.layers
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout

import models


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
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


#########################################################
# Load pre-training data
#########################################################
def load_pre_training_data(base_path=kRelationBasePath):
    CelebA = dict()
    CelebA['data'] = np.load(os.path.join(base_path, 'CeleData.npy'))
    CelebaLabel = sio.loadmat(os.path.join(base_path, 'Celeba/celebalabel.mat'))
    CelebaLabel = CelebaLabel['CelebaLabel']
    CelebA['label'] = CelebaLabel.astype('float64')

    Kaggle = dict()
    KaggleData = sio.loadmat(os.path.join(base_path, 'fer2013/mat/all/kaggleData.mat'))
    KaggleData = KaggleData['DATA']
    Kaggle['data'] = KaggleData.astype('float64')
    KaggleLabel = sio.loadmat(os.path.join(base_path, 'fer2013/mat/all/kaggleLabel.mat'))
    KaggleLabel = KaggleLabel['LABEL']
    Kaggle['label'] = KaggleLabel.astype('float64')

    # select number of 20000 samples from each data set
    pre_celeb_indices = np.arange(len(CelebA['data']))  # Get A Test Batch
    np.random.shuffle(pre_celeb_indices)
    pre_celeb_tr_indices = pre_celeb_indices[0:100000]
    pre_celeb_te_indices = pre_celeb_indices[190000:202500]

    pre_kaggle_indices = np.arange(len(Kaggle['data']))
    np.random.shuffle(pre_kaggle_indices)
    pre_kaggle_tr_indices = pre_kaggle_indices[0:30000]
    pre_kaggle_te_indices = pre_kaggle_indices[30000:35000]

    # Data = np.concatenate((CelebaData[pre_celeb_indices],KaggleData[pre_kaggle_indices]))
    # label = np.concatenate((CelebaLabel[pre_celeb_indices], KaggleLabel[pre_kaggle_indices]))

    # pre_training parameters
    pre_trX = np.concatenate((CelebA['data'][pre_celeb_tr_indices], Kaggle['data'][pre_kaggle_tr_indices]))
    pre_trY = np.concatenate((CelebA['label'][pre_celeb_tr_indices], Kaggle['label'][pre_kaggle_tr_indices]))
    pre_teX = np.concatenate((CelebA['data'][pre_celeb_te_indices], Kaggle['data'][pre_kaggle_te_indices]))
    pre_teY = np.concatenate((CelebA['label'][pre_celeb_te_indices], Kaggle['label'][pre_kaggle_te_indices]))

    pre_trX = pre_trX.reshape(-1, 48, 48, 1)  # 48x48x1 input img
    pre_teX = pre_teX.reshape(-1, 48, 48, 1)  # 48x48x1 input img

    print('load pre training data complete!!')

    return dict(trainX=pre_trX, trainY=pre_trY, testX=pre_teX, testY=pre_teY)


#########################################################
# Load relation data
#########################################################
def load_relation_data(base_path=kRelationBasePath):
    # TODO: have to figure out what data here mean
    Exp_tr1 = np.load('/home/user/Desktop/Workspace/relation/Exp_te1.npy')
    Exp_tr2 = np.load('/home/user/Desktop/Workspace/relation/Exp_te2.npy')
    Exp_te1 = np.load('/home/user/Desktop/Workspace/relation/Exp_tr1.npy')
    Exp_te2 = np.load('/home/user/Desktop/Workspace/relation/Exp_tr2.npy')

    Exp_teY = np.load('/home/user/Desktop/Workspace/relation/Exp_teY.npy')
    Exp_trY = np.load('/home/user/Desktop/Workspace/relation/Exp_trY.npy')

    Exp_trX1 = Exp_tr1.reshape(-1, 48, 48, 1)  # 48x48x1 input img
    Exp_trX2 = Exp_tr2.reshape(-1, 48, 48, 1)  # 48x48x1 input img

    Exp_teX1 = Exp_te1.reshape(-1, 48, 48, 1)  # 48x48x1 input img
    Exp_teX2 = Exp_te2.reshape(-1, 48, 48, 1)  # 48x48x1 input


#########################################################
# Load interaction data and GT
#########################################################
def load_interaction_data_and_gt():
    # TODO: have to figure out what GT here means
    GTjpg1 = sio.loadmat('./GT/Inter_group/GT2jpg1.mat')
    GTjpg2 = sio.loadmat('./GT/Inter_group/GT2jpg2.mat')
    GTjpg3 = sio.loadmat('./GT/Inter_group/GT2jpg3.mat')


#########################################################
# ### labeling face_attribute
#########################################################
def labeling_face_attribute(trainY, testY):
    # originally Pre_trY and Pre_teY
    num_train_samples = trainY.shape(1)
    num_test_samples = testY.shape(1)
    face_attribute_labels = dict(train=[], test=[])
    for model_idx in range(15):
        face_attribute_labels['train'].append(trainY[:, model_idx].reshape(num_train_samples, 1))
        face_attribute_labels['test'].append(testY[:, model_idx].reshape(num_test_samples, 1))
    return face_attribute_labels


#########################################################
# ### labeling Relation
#########################################################
def labeling_relation(trainY, testY):
    # originally Exp_trYr[1~8] and Exp_teYr[1~8]
    num_train_samples = trainY.shape(1)
    num_test_samples = testY.shape(1)
    y_for_relations = dict(train=[], test=[])
    for relation_idx in range(kNumRelations):
        y_for_relations['train'].append(trainY[:, relation_idx].reshape(num_train_samples, 1))
        y_for_relations['test'].append(testY[:, relation_idx].reshape(num_test_samples, 1))
    print ('load fine data complete!!')
    return y_for_relations


#########################################################
# MODEL CONSTRUCTION
#########################################################
def build_social_relation_network():
    deep_conv_net = build_deep_conv_net()



#########################################################
# Expression model's last layer, 8kinds
#########################################################
def build_relation_network(base_network, num_expressions=15):
    expression_models = num_expressions * [Sequential()]
    for cur_model in expression_models:
        cur_model.add(base_network)
        cur_model.add(Dense(1, activation='sigmoid'))

    return expression_models


#########################################################
# Expression model compilation
#########################################################
def set_model_optimization_scheme(models):
    model_optimizer = keras.optimizers.Adam(
        lr=1e-4, beta_1=0.99, beta_2=0.99, epsilon=1e-08, decay=1e-4)
    for cur_model in models:
        cur_model.compile(optimizer=model_optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
    return models


# #########################################################
# # Expression model fitting
# #########################################################

# Expression1.fit(x=pre_trX, y=Face_attri_trY1,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression2.fit(x=pre_trX, y=Face_attri_trY2,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression3.fit(x=pre_trX, y=Face_attri_trY3,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression4.fit(x=pre_trX, y=Face_attri_trY4,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression5.fit(x=pre_trX, y=Face_attri_trY5,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression6.fit(x=pre_trX, y=Face_attri_trY6,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression7.fit(x=pre_trX, y=Face_attri_trY7,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression8.fit(x=pre_trX, y=Face_attri_trY8,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression9.fit(x=pre_trX, y=Face_attri_trY9,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression10.fit(x=pre_trX, y=Face_attri_trY10,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression11.fit(x=pre_trX, y=Face_attri_trY11,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression12.fit(x=pre_trX, y=Face_attri_trY12,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression13.fit(x=pre_trX, y=Face_attri_trY13,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression14.fit(x=pre_trX, y=Face_attri_trY14,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])
# Expression15.fit(x=pre_trX, y=Face_attri_trY15,
#                       batch_size=120, epochs=5, verbose=1,callbacks=[tbCallBack])

# #########################################################
# # serialize Expression model save
# #########################################################
# # serialize model to JSON
# model_json = Expression_model.to_json()
# with open("Expression_model.json", "w") as json_file:
#     json_file.write(model_json)

# # serialize weights to HDF5
# Expression_model.save_weights("Expression_model.h5")
# print("Saved model to disk")


#########################################################
#  Sequential final model like sudo siamese
#########################################################
# TODO: delete last layer   <== ????

# TODO: findout what final model represents for
final_model = Sequential()
final_model.add(Dense(2048, input_shape=(4096,), activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))

final_model.add(Dense(1024, activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))

final_model.add(Dense(512, activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))

final_model.add(Dense(256, activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))

# last layers, 8 relation
relation_models = kNumRelations * [Sequential()]
for cur_model in relation_models:
    cur_model.add(final_model)
    cur_model.add(Dense(1, activation='sigmoid'))

# final_opti = keras.optimizers.Adam(lr=1e-6,  beta_1=0.99,
#                                   beta_2=0.99, epsilon=1e-08, decay=1e-4)
#
# final_model.compile(optimizer=final_opti
#                    , loss='categorical_crossentropy'
#                    , metrics=['accuracy'])
#
# final_model.fit(x=merged, y=Exp_trY,
#                batch_size=120, epochs=3000,verbose=1)
#


#########################################################
#  Sequential final model like sudo siamese
#########################################################
Face_layer_model = Model(inputs=Expression_model.input,
                         outputs=Expression_model.get_layer(index=22).output)

input_a = Input(shape=(48, 48, 1))
input_b = Input(shape=(48, 48, 1))
processed_a = Face_layer_model(input_a)
processed_b = Face_layer_model(input_b)

intermediate_layer_model = Model(inputs=Expression_model.input,
                                 outputs=Expression_model.get_layer(index=22).output)

#########################################################
#  Relatio model part, fitting & complilation
#########################################################
k1 = Face_layer_model.predict(Exp_trX1)
k2 = Face_layer_model.predict(Exp_trX2)

merged = np.concatenate((k1, k2), axis=1)

relation_model_optimizer = keras.optimizers.Adam(
    lr=1e-4, beta_1=0.99, beta_2=0.99, epsilon=1e-08, decay=1e-4)

for cur_model in relation_models:
    cur_model.compile(optimizer=relation_model_optimizer,
                      loos='binary_crossentropy',
                      metrics=['accuracy'])
    # TODO: set y to an appropriate set
    cur_model.fit(x=merged, y=Exp_trYr_i, batch_size=120, epochs=400, verbose=1)


# #########################################################
# #  Relation model evaluation
# #########################################################
# k1 = intermediate_layer_model.predict(Exp_teX1)
# k2 = intermediate_layer_model.predict(Exp_teX2)

# merged = np.concatenate((k1,k2),axis=1)


# score = Relation1_model.evaluate(merged, Exp_teYr1, verbose=0)
# print('1 Test score:', score[0])
# print('1 Test accuracy:', score[1])

# score = Relation2_model.evaluate(merged, Exp_teYr2,  verbose=0)
# print('2 Test score:', score[0])
# print('2 Test accuracy:', score[1])

# score = Relation3_model.evaluate(merged, Exp_teYr3,  verbose=0)
# print('3 Test score:', score[0])
# print('3 Test accuracy:', score[1])

# score = Relation4_model.evaluate(merged, Exp_teYr4,  verbose=0)
# print('4 Test score:', score[0])
# print('4 Test accuracy:', score[1])

# score = Relation5_model.evaluate(merged, Exp_teYr5, verbose=0)
# print('5 Test score:', score[0])
# print('5 Test accuracy:', score[1])

# score = Relation6_model.evaluate(merged, Exp_teYr6,  verbose=0)
# print('6 Test score:', score[0])
# print('6 Test accuracy:', score[1])

# score = Relation7_model.evaluate(merged, Exp_teYr7,  verbose=0)
# print('7 Test score:', score[0])
# print('7 Test accuracy:', score[1])

# score = Relation8_model.evaluate(merged, Exp_teYr8, verbose=0)
# print('8 Test score:', score[0])
# print('8 Test accuracy:', score[1])


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


if __name__ == "__main__":





#()()
#('')HAANJU.YOO
