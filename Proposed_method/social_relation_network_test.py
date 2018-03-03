from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge, Input
from keras.layers import merge
from keras.layers.core import Dropout
from keras.models import Model
from keras.models import model_from_json

from keras.utils import np_utils
import scipy.io as sio
import numpy as np
import keras.layers
import os

from keras.utils.vis_utils import model_to_dot

print('import packages complete')

#########################################################
# Pre-defines
#########################################################
kNumRelations = 8
kRelationBasePath = '/home/user/Desktop/Workspace/relation'
kInteractionBasePath = '/home/user/Desktop/Workspace/Interaction_relation'
kHeadPairBasePath = '/home/user/Workspace/Interaction_relation/DATA/new/relation_case'


#########################################################
# PART. social relation predicting for grouping
#########################################################
def load_models(base_path=kInteractionBasePath):
    # load final model
    final_model_json_file = open(os.path.join(base_path, 'final_model_json.json'), 'r')
    final_model_json = final_model_json_file.read()
    final_model_json_file.close()
    final_model = model_from_json(final_model_json)
    final_model.load_weights(os.path.join(base_path, 'final_model.h5'))

    # expression model
    exp_model_json_file = open(os.path.join(base_path, ''))

    relation_models = []
    for model_id in range(kNumRelations):
        model_json_file = open(os.path.join(base_path, 'Relation%d_model.json' % model_id))
        model_json = model_json_file.read()
        model_json_file.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(os.path.join(base_path, 'Relation%d_model.h5' % model_id))
        relation_models.append(loaded_model)

    print("Loaded model from disk")
    return final_model, relation_models


def load_head_pairs(path=kHeadPairBasePath):
    file_name_list = os.listdir(path)

idx = len(list_name)
file_format = '_relation.mat'

for i in range(idx):
    temp_name = list_name[i]
    temp_ind = temp_name.split('_')
    temp_relation = '/home/user/Workspace/Interaction_relation/DATA/new/relation_case/' + temp_ind[0] + file_format
    temp_raw_relation = sio.loadmat(temp_relation)

    temp_croped_head1 = temp_raw_relation['croped_head1']
    temp_croped_head2 = temp_raw_relation['croped_head2']

    k1 = Face_layer_model.predict(temp_croped_head1)
    k2 = Face_layer_model.predict(temp_croped_head2)

    merged = np.concatenate((k1, k2), axis=1)

    score1 = Relation1_model.predict(merged, verbose=0)
    score2 = Relation2_model.predict(merged, verbose=0)
    score3 = Relation3_model.predict(merged, verbose=0)
    score4 = Relation4_model.predict(merged, verbose=0)
    score5 = Relation5_model.predict(merged, verbose=0)
    score6 = Relation6_model.predict(merged, verbose=0)
    score7 = Relation7_model.predict(merged, verbose=0)
    score8 = Relation8_model.predict(merged, verbose=0)