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
#from keras.utils import Sequence, multi_gpu_model

import random
import sys
from multiprocessing import cpu_count
import numpy as np
import glob
from skimage.io import imread
import cv2
from Smarthome_Loader import *

#epochs = int(sys.argv[1])
#model_name = sys.argv[2]
num_classes = 35
batch_size = 16
stack_size = 64

train_generator = DataLoader_video_train('/data/stars/user/sdas/smarthomes_data/splits/train_CS.txt', batch_size = batch_size)
#val_generator = DataLoader_video_train('/data/stars/user/sdas/NTU_RGB/splits/imp_files/validation_new.txt', batch_size = batch_size)
test_generator = DataLoader_video_test('/data/stars/user/sdas/smarthomes_data/splits/test_CS.txt', batch_size = batch_size)

