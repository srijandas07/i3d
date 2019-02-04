import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.utils import Sequence, to_categorical

import numpy as np
from random import sample, randint, shuffle
import glob
import cv2
import time

class DataLoader_video_train(Sequence):
    def __init__(self, path1, version, batch_size = 4):
        self.batch_size = batch_size
        #
        self.version = version
        self.path = '/data/stars/user/rdai/smarthomes/'+self.version+'/'
        
        self.files = [i.strip() for i in open(path1).readlines()]
        #
        self.stack_size = 64    
        #
        self.num_classes = 19

        self.stride = 2

    def _name_to_int(self,name):
        integer=0
        if name=="Cutbread":
            integer=1
        elif name=="Drink.Frombottle":
            integer=2
        elif name=="Drink.Fromcan":
            integer=3
        elif name=="Drink.Fromcup":
            integer=4
        elif name=="Drink.Fromglass":
            integer=5
        elif name=="Eat.Attable":
            integer=6
        elif name=="Eat.Snack":
            integer=7
        elif name=="Enter":
            integer=8
        elif name=="Getup":
            integer=9
        elif name=="Leave":
            integer=10
        elif name=="Pour.Frombottle":
            integer=11
        elif name=="Pour.Fromcan":
            integer=12
        elif name=="Readbook":
            integer=13
        elif name=="Sitdown":
            integer=14
        elif name=="Takepills":
            integer=15
        elif name=="Uselaptop":
            integer=16
        elif name=="Usetablet":
            integer=17
        elif name=="Usetelephone":
            integer=18
        elif name=="Walk":
            integer=19
        return integer

    def __len__(self):
        return int(len(self.files) / self.batch_size)
    
    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = [os.path.splitext(i)[0] for i in batch]
        x_train = [self._get_video(i) for i in batch]
        x_train = np.array(x_train, np.float32)

        #
        x_train /= 127.5
        x_train -= 1
        #y_train = np.array([int(i[-3:]) for i in batch]) - 1
        y_train = np.array([self._name_to_int(i.split('_')[0]) for i in batch]) - 1
        y_train = to_categorical(y_train, num_classes = self.num_classes)

        return x_train, y_train

    def _get_video(self, vid_name):
        images = glob.glob(self.path + vid_name + "/*")
        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])
            
        files.sort()
        
        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])

        return arr

    def on_epoch_end(self):
        shuffle(self.files)



class DataLoader_video_test(Sequence):
    def __init__(self, path1, version, batch_size = 4):
        self.batch_size = batch_size
        #
        self.version = version
        self.path = '/data/stars/user/rdai/smarthomes/'+self.version+'/'
        
        self.files = [i.strip() for i in open(path1).readlines()]
        #
        self.stack_size = 64    
        #
        self.num_classes = 19
        self.stride = 2

    def _name_to_int(self,name):
        integer=0
        if name=="Cutbread":
            integer=1
        elif name=="Drink.Frombottle":
            integer=2
        elif name=="Drink.Fromcan":
            integer=3
        elif name=="Drink.Fromcup":
            integer=4
        elif name=="Drink.Fromglass":
            integer=5
        elif name=="Eat.Attable":
            integer=6
        elif name=="Eat.Snack":
            integer=7
        elif name=="Enter":
            integer=8
        elif name=="Getup":
            integer=9
        elif name=="Leave":
            integer=10
        elif name=="Pour.Frombottle":
            integer=11
        elif name=="Pour.Fromcan":
            integer=12
        elif name=="Readbook":
            integer=13
        elif name=="Sitdown":
            integer=14
        elif name=="Takepills":
            integer=15
        elif name=="Uselaptop":
            integer=16
        elif name=="Usetablet":
            integer=17
        elif name=="Usetelephone":
            integer=18
        elif name=="Walk":
            integer=19
        return integer

    def __len__(self):
        return int(len(self.files) / self.batch_size)
    
    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = [os.path.splitext(i)[0] for i in batch]
        x_train = [self._get_video(i) for i in batch]
        x_train = np.array(x_train, np.float32)

        #
        x_train /= 127.5
        x_train -= 1
        #y_train = np.array([int(i[-3:]) for i in batch]) - 1
        y_train = np.array([self._name_to_int(i.split('_')[0]) for i in batch]) - 1
        y_train = to_categorical(y_train, num_classes = self.num_classes)

        return x_train, y_train

    def _get_video(self, vid_name):
        images = glob.glob(self.path + vid_name + "/*")
        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])
            
        files.sort()
        
        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])

        return arr

    def on_epoch_end(self):
        shuffle(self.files)

