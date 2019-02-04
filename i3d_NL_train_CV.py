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
from Smarthome_Loader_CV import *
from non_local import non_local_block
from keras.layers import AveragePooling3D

epochs = int(sys.argv[1])
model_name = sys.argv[2]
version = sys.argv[3]
num_classes = 19
batch_size = 16
stack_size = 64

class i3d_modified:
    def __init__(self, weights = 'rgb_imagenet_and_kinetics'):
        self.model = Inception_Inflated3d(include_top = True, weights= weights)
        
    def i3d_flattened(self, num_classes = 19):
        i3d = Model(inputs = self.model.input, outputs = self.model.get_layer(index=-4).output)
        x = conv3d_bn(i3d.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Flatten()(x)
        predictions = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
        new_model  = Model(inputs = i3d.input, outputs = predictions)
        
        #for layer in i3d.layers:
        #    layer.trainable = False
        return new_model

class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):

        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')


i3d = i3d_modified(weights = 'rgb_imagenet_and_kinetics')
model_branch = i3d.i3d_flattened(num_classes = num_classes)
#model_branch.load_weights('/data/stars/user/sdas/PhD_work/ICCV_2019/models/epoch_full_body_NTU_CS.hdf5')
model_branch.load_weights('/data/stars/user/rdai/smarthomes/i3d/i3d/weights_Blurred_i3d_without_window_cross_view/epoch_30.hdf5')
model_i3d = Model(inputs = model_branch.input, outputs = model_branch.get_layer('Mixed_5c').output)
x = non_local_block(model_i3d.output, compression=2, mode='embedded')
#x = non_local_block(x, compression=2, mode='embedded')
#x = non_local_block(x, compression=2, mode='embedded')
#x = non_local_block(x, compression=2, mode='embedded')
#x = non_local_block(x, compression=2, mode='embedded')
x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool'+'second')(x)
x = Dropout(0.0)(x)
x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1'+'second')
x = Flatten(name='flatten'+'second')(x)
predictions = Dense(num_classes, activation='softmax', name='softmax'+'second')(x)
model = Model(inputs=model_branch.input, outputs=predictions, name = 'i3d_nonlocal')
optim = SGD(lr = 0.01, momentum = 0.9)
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
for l_m, l_lh in zip(model.layers[-5: -4], model_branch.layers[-5: -4]):
    l_m.set_weights(l_lh.get_weights())
    l_m.trainable = True

#model = load_model("../weights3/epoch11.hdf5")
# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 10)
#filepath = '../weights3/weights.{epoch:04d}-{val_loss:.2f}.hdf5'
csvlogger = CSVLogger(model_name+'_ntu.csv')

parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])

model_checkpoint = CustomModelCheckpoint(model, './weights_'+model_name+'/epoch_')
#model_checkpoint = ModelCheckpoint('./weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5')

train_generator = DataLoader_video_train('/data/stars/user/sdas/smarthomes_data/splits/train_CV.txt',version, batch_size = batch_size)
val_generator = DataLoader_video_train('/data/stars/user/sdas/smarthomes_data/splits/validation_CV.txt', version, batch_size = batch_size)
test_generator = DataLoader_video_test('/data/stars/user/sdas/smarthomes_data/splits/test_CV.txt', version, batch_size = batch_size)

parallel_model.fit_generator(
    generator = train_generator, 
    epochs = epochs, 
    callbacks = [csvlogger, reduce_lr, model_checkpoint], 
    validation_data=val_generator,
    max_queue_size = 48,
    workers = cpu_count() - 2,
    use_multiprocessing = False,
)

#print(parallel_model.evaluate_generator(generator = test_generator))

model_result=parallel_model.evaluate_generator(generator = test_generator)
print(model_result)

with open(model_name+"_CV_result.txt", "a+") as f:
    f.write(str(model_result)+"\n")