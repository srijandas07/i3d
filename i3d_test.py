import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from keras.layers import Dense, Flatten, Dropout, Reshape
from keras import regularizers
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.optimizers import SGD
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import Sequence, multi_gpu_model

import random
import sys
from multiprocessing import cpu_count
import numpy as np
import glob
from skimage.io import imread
import cv2
from Smarthome_Loader import *


num_classes = 35
batch_size = 16
stack_size = 64

class i3d_modified:
    def __init__(self, weights = 'rgb_imagenet_and_kinetics'):
        self.model = Inception_Inflated3d(include_top = True, weights= weights)
        
    def i3d_flattened(self, num_classes = 35):
        i3d = Model(inputs = self.model.input, outputs = self.model.get_layer(index=-4).output)
        x = conv3d_bn(i3d.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Flatten()(x)
        predictions = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
        new_model  = Model(inputs = i3d.input, outputs = predictions)        
        #for layer in i3d.layers:
        #    layer.trainable = False
        return new_model


i3d = i3d_modified(weights = 'rgb_imagenet_and_kinetics')
model = i3d.i3d_flattened(num_classes = num_classes)
optim = SGD(lr = 0.01, momentum = 0.9)

model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
model.load_weights('./weights_i3d_without_window/epoch_29.hdf5')
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])

test_generator = DataLoader_video_test('/data/stars/user/sdas/smarthomes_data/splits/test_CS.txt', 'smarthome_clipped_SSD', batch_size = batch_size)


print(parallel_model.evaluate_generator(generator = test_generator, use_multiprocessing=True, workers=cpu_count()-2))


