
# # Relation for Interaction group Recognition
# ## Ver. Keras
# ## 2017.04.05


#########################################################
# ## Load package
#########################################################


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

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


print('import packages complete')

#########################################################
# ## tensorboard setup
#########################################################
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


# #########################################################
# # ### Load Pre_Training data
# #########################################################

# CelebaData = np.load('/home/user/Desktop/Workspace/relation/CeleData.npy')
# KaggleData = sio.loadmat('/home/user/Desktop/Workspace/relation/fer2013/mat/all/kaggleData.mat')
# KaggleData = KaggleData['DATA']
# KaggleData = KaggleData.astype('float64')

# CelebaLabel = sio.loadmat('/home/user/Desktop/Workspace/relation/Celeba/celebalabel.mat')
# CelebaLabel = CelebaLabel['CelebaLabel']
# CelebaLabel = CelebaLabel.astype('float64')

# KaggleLabel = sio.loadmat('/home/user/Desktop/Workspace/relation/fer2013/mat/all/kaggleLabel.mat')
# KaggleLabel = KaggleLabel['LABEL']
# KaggleLabel = KaggleLabel.astype('float64')

# #select number of 20000 samples from each data set
# pre_celeb_indices = np.arange(len(CelebaData)) # Get A Test Batch
# np.random.shuffle(pre_celeb_indices)
# pre_celeb_tr_indices = pre_celeb_indices[0:100000]
# pre_celeb_te_indices = pre_celeb_indices[190000:202500]

# pre_kaggle_indices = np.arange(len(KaggleData))
# np.random.shuffle(pre_kaggle_indices)
# pre_kaggle_tr_indices = pre_kaggle_indices[0:30000]
# pre_kaggle_te_indices = pre_kaggle_indices[30000:35000]

# # Data = np.concatenate((CelebaData[pre_celeb_indices],KaggleData[pre_kaggle_indices]))
# # label = np.concatenate((CelebaLabel[pre_celeb_indices], KaggleLabel[pre_kaggle_indices]))

# # pre_training parameters
# pre_trX = np.concatenate((CelebaData[pre_celeb_tr_indices],KaggleData[pre_kaggle_tr_indices]))
# pre_trY = np.concatenate((CelebaLabel[pre_celeb_tr_indices],KaggleLabel[pre_kaggle_tr_indices]))
# pre_teX = np.concatenate((CelebaData[pre_celeb_te_indices],KaggleData[pre_kaggle_te_indices]))
# pre_teY = np.concatenate((CelebaLabel[pre_celeb_te_indices],KaggleLabel[pre_kaggle_te_indices]))

# pre_trX = pre_trX.reshape(-1, 48, 48, 1)  # 48x48x1 input img
# pre_teX = pre_teX.reshape(-1, 48, 48, 1)  # 48x48x1 input img

# print ('load pre training data complete!!')

# #########################################################
# # ### Load Relation data
# #########################################################
# Exp_tr1 = np.load('/home/user/Desktop/Workspace/relation/Exp_te1.npy')
# Exp_tr2 = np.load('/home/user/Desktop/Workspace/relation/Exp_te2.npy')
# Exp_te1 = np.load('/home/user/Desktop/Workspace/relation/Exp_tr1.npy')
# Exp_te2 = np.load('/home/user/Desktop/Workspace/relation/Exp_tr2.npy')

# Exp_teY = np.load('/home/user/Desktop/Workspace/relation/Exp_teY.npy')
# Exp_trY = np.load('/home/user/Desktop/Workspace/relation/Exp_trY.npy')

# Exp_trX1 = Exp_tr1.reshape(-1, 48, 48, 1)  # 48x48x1 input img
# Exp_trX2 = Exp_tr2.reshape(-1, 48, 48, 1)  # 48x48x1 input img

# Exp_teX1 = Exp_te1.reshape(-1, 48, 48, 1)  # 48x48x1 input img
# Exp_teX2 = Exp_te2.reshape(-1, 48, 48, 1)  # 48x48x1 input 



# #########################################################
# # ## Load Interaction data, GT
# #########################################################
# GTjpg1 = sio.loadmat('./GT/Inter_group/GT2jpg1.mat')
# GTjpg2 = sio.loadmat('./GT/Inter_group/GT2jpg2.mat')
# GTjpg3 = sio.loadmat('./GT/Inter_group/GT2jpg3.mat')


# #########################################################
# # ### labeling face_attribute
# #########################################################
# [r3,c3] = pre_trY.shape
# Face_attri_trY1 = pre_trY[:,0].reshape(r3,1)
# Face_attri_trY2 = pre_trY[:,1].reshape(r3,1)
# Face_attri_trY3 = pre_trY[:,2].reshape(r3,1)
# Face_attri_trY4 = pre_trY[:,3].reshape(r3,1)
# Face_attri_trY5 = pre_trY[:,4].reshape(r3,1)
# Face_attri_trY6 = pre_trY[:,5].reshape(r3,1)
# Face_attri_trY7 = pre_trY[:,6].reshape(r3,1)
# Face_attri_trY8 = pre_trY[:,7].reshape(r3,1)
# Face_attri_trY9 = pre_trY[:,8].reshape(r3,1)
# Face_attri_trY10 = pre_trY[:,9].reshape(r3,1)
# Face_attri_trY11 = pre_trY[:,10].reshape(r3,1)
# Face_attri_trY12 = pre_trY[:,11].reshape(r3,1)
# Face_attri_trY13 = pre_trY[:,12].reshape(r3,1)
# Face_attri_trY14 = pre_trY[:,13].reshape(r3,1)
# Face_attri_trY15 = pre_trY[:,14].reshape(r3,1)

# [r4,c4] = pre_teY.shape
# Face_attri_teY1 = pre_teY[:,0].reshape(r4,1)
# Face_attri_teY2 = pre_teY[:,1].reshape(r4,1)
# Face_attri_teY3 = pre_teY[:,2].reshape(r4,1)
# Face_attri_teY4 = pre_teY[:,3].reshape(r4,1)
# Face_attri_teY5 = pre_teY[:,4].reshape(r4,1)
# Face_attri_teY6 = pre_teY[:,5].reshape(r4,1)
# Face_attri_teY7 = pre_teY[:,6].reshape(r4,1)
# Face_attri_teY8 = pre_teY[:,7].reshape(r4,1)
# Face_attri_teY9 = pre_teY[:,8].reshape(r4,1)
# Face_attri_teY10 = pre_teY[:,9].reshape(r4,1)
# Face_attri_teY11 = pre_teY[:,10].reshape(r4,1)
# Face_attri_teY12 = pre_teY[:,11].reshape(r4,1)
# Face_attri_teY13 = pre_teY[:,12].reshape(r4,1)
# Face_attri_teY14 = pre_teY[:,13].reshape(r4,1)
# Face_attri_teY15 = pre_teY[:,14].reshape(r4,1)

# #########################################################
# # ### labeling Relation
# #########################################################

# [r1,c1] = Exp_trY.shape
# Exp_trYr1 = Exp_trY[:,0].reshape(r1,1)
# Exp_trYr2 = Exp_trY[:,1].reshape(r1,1)
# Exp_trYr3 = Exp_trY[:,2].reshape(r1,1)
# Exp_trYr4 = Exp_trY[:,3].reshape(r1,1)
# Exp_trYr5 = Exp_trY[:,4].reshape(r1,1)
# Exp_trYr6 = Exp_trY[:,5].reshape(r1,1)
# Exp_trYr7 = Exp_trY[:,6].reshape(r1,1)
# Exp_trYr8 = Exp_trY[:,7].reshape(r1,1)


# [r2,c2] = Exp_teY.shape
# Exp_teYr1 = Exp_teY[:,0].reshape(r2,1)
# Exp_teYr2 = Exp_teY[:,1].reshape(r2,1)
# Exp_teYr3 = Exp_teY[:,2].reshape(r2,1)
# Exp_teYr4 = Exp_teY[:,3].reshape(r2,1)
# Exp_teYr5 = Exp_teY[:,4].reshape(r2,1)
# Exp_teYr6 = Exp_teY[:,5].reshape(r2,1)
# Exp_teYr7 = Exp_teY[:,6].reshape(r2,1)
# Exp_teYr8 = Exp_teY[:,7].reshape(r2,1)

# print ('load fine data complete!!')


#########################################################
# Expression Sequential model 
#########################################################

Expression_model = Sequential()

#layer 1
Expression_model.add(Convolution2D(1,5,5, border_mode='same'
                        ,input_shape=(48,48,1),activation='relu'))
Expression_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
Expression_model.add(BatchNormalization())
Expression_model.add(Dropout(0.5))

#layer 2
Expression_model.add((Convolution2D(64,5,5,border_mode='same'
                         ,activation='relu',)))
Expression_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
Expression_model.add(BatchNormalization())
Expression_model.add(Dropout(0.5))


#layer 4
Expression_model.add((Convolution2D(96,5,5,border_mode='same'
                         ,activation='relu',)))
Expression_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
Expression_model.add(BatchNormalization())
Expression_model.add(Dropout(0.5))

#layer 5
Expression_model.add((Convolution2D(256,5,5,border_mode='same'
                         ,activation='relu',)))
Expression_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
Expression_model.add(BatchNormalization())
Expression_model.add(Dropout(0.5))

#layer 6
Expression_model.add((Convolution2D(256,5,5,border_mode='same'
                         ,activation='relu',)))
Expression_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
Expression_model.add(BatchNormalization())
Expression_model.add(Dropout(0.5))
Expression_model.add(Flatten())

#layer 7
Expression_model.add(Dense(2048,activation='relu'))
Expression_model.add(BatchNormalization())
Expression_model.add(Dropout(0.5))


#########################################################
# Expression model's last layer, 8kinds
#########################################################

Expression1 = Sequential()
Expression1.add(Expression_model)
Expression1.add(Dense(1,activation='sigmoid'))

Expression2 = Sequential()
Expression2.add(Expression_model)
Expression2.add(Dense(1,activation='sigmoid'))

Expression3 = Sequential()
Expression3.add(Expression_model)
Expression3.add(Dense(1,activation='sigmoid'))


Expression4 = Sequential()
Expression4.add(Expression_model)
Expression4.add(Dense(1,activation='sigmoid'))


Expression5 = Sequential()
Expression5.add(Expression_model)
Expression5.add(Dense(1,activation='sigmoid'))


Expression6 = Sequential()
Expression6.add(Expression_model)
Expression6.add(Dense(1,activation='sigmoid'))


Expression7 = Sequential()
Expression7.add(Expression_model)
Expression7.add(Dense(1,activation='sigmoid'))


Expression8 = Sequential()
Expression8.add(Expression_model)
Expression8.add(Dense(1,activation='sigmoid'))


Expression9 = Sequential()
Expression9.add(Expression_model)
Expression9.add(Dense(1,activation='sigmoid'))


Expression10 = Sequential()
Expression10.add(Expression_model)
Expression10.add(Dense(1,activation='sigmoid'))


Expression11 = Sequential()
Expression11.add(Expression_model)
Expression11.add(Dense(1,activation='sigmoid'))

Expression12 = Sequential()
Expression12.add(Expression_model)
Expression12.add(Dense(1,activation='sigmoid'))

Expression13 = Sequential()
Expression13.add(Expression_model)
Expression13.add(Dense(1,activation='sigmoid'))

Expression14 = Sequential()
Expression14.add(Expression_model)
Expression14.add(Dense(1,activation='sigmoid'))

Expression15 = Sequential()
Expression15.add(Expression_model)
Expression15.add(Dense(1,activation='sigmoid'))



#########################################################
# Expression model optimization setting
#########################################################

Expression_opti = keras.optimizers.Adam(lr=1e-4,  beta_1=0.99,
                                   beta_2=0.99, epsilon=1e-08, decay=1e-4)

#Expression_model.compile(optimizer=Expression_opti
#                    , loss='categorical_crossentropy'
#                    , metrics=['accuracy'])
#
#Expression_model.fit(x=pre_trX, y=pre_trY,
#                batch_size=120, epochs=400,verbose=1)

#########################################################
# Expression model compilation
#########################################################

Expression1.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression2.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression3.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression4.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression5.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression6.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression7.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression8.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression9.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression10.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression11.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression12.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression13.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression14.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])
Expression15.compile(optimizer=Expression_opti
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])

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
#  delete last layer

## model setting
final_model = Sequential()
final_model.add(Dense(2048,input_shape=(4096,), activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))

final_model.add(Dense(1024,activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))

final_model.add(Dense(512,activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))

final_model.add(Dense(256, activation='relu'))
final_model.add(BatchNormalization())
final_model.add(Dropout(0.5))

# last layers, 8 relation
Relation1_model = Sequential()
Relation1_model.add(final_model)
Relation1_model.add(Dense(1, activation='sigmoid'))

Relation2_model = Sequential()
Relation2_model.add(final_model)
Relation2_model.add(Dense(1, activation='sigmoid'))

Relation3_model = Sequential()
Relation3_model.add(final_model)
Relation3_model.add(Dense(1, activation='sigmoid'))

Relation4_model = Sequential()
Relation4_model.add(final_model)
Relation4_model.add(Dense(1, activation='sigmoid'))

Relation5_model = Sequential()
Relation5_model.add(final_model)
Relation5_model.add(Dense(1, activation='sigmoid'))

Relation6_model = Sequential()
Relation6_model.add(final_model)
Relation6_model.add(Dense(1, activation='sigmoid'))

Relation7_model = Sequential()
Relation7_model.add(final_model)
Relation7_model.add(Dense(1, activation='sigmoid'))

Relation8_model = Sequential()
Relation8_model.add(final_model)
Relation8_model.add(Dense(1, activation='sigmoid'))

#final_opti = keras.optimizers.Adam(lr=1e-6,  beta_1=0.99,
#                                   beta_2=0.99, epsilon=1e-08, decay=1e-4)
#
#final_model.compile(optimizer=final_opti
#                    , loss='categorical_crossentropy'
#                    , metrics=['accuracy'])
#
#final_model.fit(x=merged, y=Exp_trY,
#                batch_size=120, epochs=3000,verbose=1)
#


#########################################################
#  Sequential final model like sudo siamese
#########################################################
Face_layer_model = Model(inputs=Expression_model.input,
                         outputs=Expression_model.get_layer(index=22).output)


input_a = Input(shape=(48,48,1))
input_b = Input(shape=(48,48,1))
processed_a = Face_layer_model(input_a)
processed_b = Face_layer_model(input_b)

intermediate_layer_model = Model(inputs=Expression_model.input,
                         outputs=Expression_model.get_layer(index=22).output)


#########################################################
#  Relatio model part, fitting & complilation
#########################################################
k1 = Face_layer_model.predict(Exp_trX1)
k2 = Face_layer_model.predict(Exp_trX2)

merged = np.concatenate((k1,k2),axis=1)


Relation_opti = keras.optimizers.Adam(lr=1e-4,  beta_1=0.99,
                                   beta_2=0.99, epsilon=1e-08, decay=1e-4)

Relation1_model.compile(optimizer=Relation_opti
                    , loss='binary_crossentropy'
                    , metrics=['accuracy'])

# Relation1_model.fit(x=merged, y=Exp_trYr1,
#                 batch_size=120, epochs=400,verbose=1)


Relation2_model.compile(optimizer=Relation_opti
                    , loss='binary_crossentropy'
                    , metrics=['accuracy'])

# Relation2_model.fit(x=merged, y=Exp_trYr2,
#                 batch_size=120, epochs=400,verbose=1)


Relation3_model.compile(optimizer=Relation_opti
                    , loss='binary_crossentropy'
                    , metrics=['accuracy'])

# Relation3_model.fit(x=merged, y=Exp_trYr3,
#                 batch_size=120, epochs=400,verbose=1)


Relation4_model.compile(optimizer=Relation_opti
                    , loss='binary_crossentropy'
                    , metrics=['accuracy'])

# Relation4_model.fit(x=merged, y=Exp_trYr4,
#                 batch_size=120, epochs=400,verbose=1)


Relation5_model.compile(optimizer=Relation_opti
                    , loss='binary_crossentropy'
                    , metrics=['accuracy'])

# Relation5_model.fit(x=merged, y=Exp_trYr5,
#                 batch_size=120, epochs=400,verbose=1)


Relation6_model.compile(optimizer=Relation_opti
                    , loss='binary_crossentropy'
                    , metrics=['accuracy'])

# Relation6_model.fit(x=merged, y=Exp_trYr6,
#                 batch_size=120, epochs=400,verbose=1)


Relation7_model.compile(optimizer=Relation_opti
                    , loss='binary_crossentropy'
                    , metrics=['accuracy'])

# Relation7_model.fit(x=merged, y=Exp_trYr7,
#                 batch_size=120, epochs=400,verbose=1)


Relation8_model.compile(optimizer=Relation_opti
                    , loss='binary_crossentropy'
                    , metrics=['accuracy'])

# Relation8_model.fit(x=merged, y=Exp_trYr8,
#                 batch_size=120, epochs=400,verbose=1)

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


# #########################################################
# # serialize Final model save
# #########################################################
# # serialize model to JSON
# final_model_json = final_model.to_json()
# with open("final_model_json.json", "w") as json_file:
#     json_file.write(final_model_json)

# # serialize weights to HDF5
# final_model.save_weights("final_model.h5")
# print("Saved final_model model to disk")

# #########################################################
# # serialize model to JSON
# Relation1_model_json = Relation1_model.to_json()
# with open("Relation1_model.json", "w") as json_file:
#     json_file.write(Relation1_model_json)

# # serialize weights to HDF5
# Relation1_model.save_weights("Relation1_model.h5")
# print("Saved Relation1_model to disk")

# #########################################################
# # serialize model to JSON
# Relation2_model_json = Relation2_model.to_json()
# with open("Relation2_model.json", "w") as json_file:
#     json_file.write(Relation2_model_json)

# # serialize weights to HDF5
# Relation2_model.save_weights("Relation2_model.h5")
# print("Saved Relation2_model to disk")

# #########################################################
# # serialize model to JSON
# Relation3_model_json = Relation3_model.to_json()
# with open("Relation3_model.json", "w") as json_file:
#     json_file.write(Relation3_model_json)

# # serialize weights to HDF5
# Relation3_model.save_weights("Relation3_model.h5")
# print("Saved Relation3_model to disk")

# #########################################################
# # serialize model to JSON
# Relation4_model_json = Relation4_model.to_json()
# with open("Relation4_model.json", "w") as json_file:
#     json_file.write(Relation4_model_json)

# # serialize weights to HDF5
# Relation4_model.save_weights("Relation4_model.h5")
# print("Saved Relation4_model to disk")

# #########################################################
# # serialize model to JSON
# Relation5_model_json = Relation5_model.to_json()
# with open("Relation5_model.json", "w") as json_file:
#     json_file.write(Relation5_model_json)

# # serialize weights to HDF5
# Relation5_model.save_weights("Relation5_model.h5")
# print("Saved Relation5_model to disk")

# #########################################################
# # serialize model to JSON
# Relation6_model_json = Relation6_model.to_json()
# with open("Relation6_model.json", "w") as json_file:
#     json_file.write(Relation6_model_json)

# # serialize weights to HDF5
# Relation6_model.save_weights("Relation6_model.h5")
# print("Saved Relation6_model to disk")

# #########################################################
# # serialize model to JSON
# Relation7_model_json = Relation7_model.to_json()
# with open("Relation7_model.json", "w") as json_file:
#     json_file.write(Relation7_model_json)

# # serialize weights to HDF5
# Relation7_model.save_weights("Relation7_model.h5")
# print("Saved Relation7_model to disk")

# #########################################################
# # serialize model to JSON
# Relation8_model_json = Relation8_model.to_json()
# with open("Relation8_model.json", "w") as json_file:
#     json_file.write(Relation8_model_json)

# # serialize weights to HDF5
# Relation8_model.save_weights("Relation8_model.h5")
# print("Saved Relation8_model to disk")

# #########################################################
# # serialize model to JSON
# Expression_model_json = Expression_model.to_json()
# with open("Expression_model.json", "w") as json_file:
#     json_file.write(Expression_model_json)

# # serialize weights to HDF5
# Expression_model.save_weights("Expression_model.h5")
# print("Saved Expression_model to disk")

#########################################################
# PART. social relation predicting for grouping
#########################################################

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/final_model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_final_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_final_model.load_weights("/home/user/Workspace/Interaction_relation/final_model.h5")

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/Relation1_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_Relation1_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_Relation1_model.load_weights("/home/user/Workspace/Interaction_relation/Relation1_model.h5")

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/Relation2_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_Relation2_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_Relation2_model.load_weights("/home/user/Workspace/Interaction_relation/Relation2_model.h5")

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/Relation3_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_Relation3_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_Relation3_model.load_weights("/home/user/Workspace/Interaction_relation/Relation3_model.h5")

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/Relation4_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_Relation4_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_Relation4_model.load_weights("/home/user/Workspace/Interaction_relation/Relation4_model.h5")

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/Relation5_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_Relation5_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_Relation5_model.load_weights("/home/user/Workspace/Interaction_relation/Relation5_model.h5")

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/Relation6_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_Relation6_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_Relation6_model.load_weights("/home/user/Workspace/Interaction_relation/Relation6_model.h5")

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/Relation7_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_Relation7_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_Relation7_model.load_weights("/home/user/Workspace/Interaction_relation/Relation7_model.h5")

# load json and create model
json_file = open('/home/user/Workspace/Interaction_relation/Relation8_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_Relation8_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_Relation8_model.load_weights("/home/user/Workspace/Interaction_relation/Relation8_model.h5")


print("Loaded model from disk")



list_name = os.listdir("/home/user/Workspace/Interaction_relation/DATA/new/relation_case")

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

    merged = np.concatenate((k1,k2),axis=1)

    score1 = Relation1_model.predict(merged, verbose=0)
    score2 = Relation2_model.predict(merged, verbose=0)
    score3 = Relation3_model.predict(merged, verbose=0)
    score4 = Relation4_model.predict(merged, verbose=0)
    score5 = Relation5_model.predict(merged, verbose=0)
    score6 = Relation6_model.predict(merged, verbose=0)
    score7 = Relation7_model.predict(merged, verbose=0)
    score8 = Relation8_model.predict(merged, verbose=0)
 








