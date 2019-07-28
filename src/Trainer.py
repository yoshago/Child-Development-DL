import pickle
import numpy as np
from keras.models import Sequential
from random import shuffle, random

class Trainer:

    def __init__(self,path, num_classes):
        self.num_classes=num_classes
        self.cntr = np.zeros((num_classes))
        self.flags = np.full((num_classes), True, dtype=bool)
        self.path = path
        self.x_train = []
        self.y_train = []
        self.model = Sequential()
    # The dictionary is a 2d array, each of the 4 rows in this array contains integers. each integer is of the form FFSS.
    # while FF represent file number (we hava 95 files), and SS represent one sample in this file.
    #All the samples in each file is belong to same class.
    def load_dictionary(self, dict_name):
        with open(self.path+dict_name,'rb') as data_key:
            self.dictionary = np.array(pickle.load(data_key))
        for i in range(len(self.dictionary)):
                shuffle(self.dictionary[i])

    # thre is 95 files of data, each of them divided for samples. the data_key array (from above) contains locations and classifications).
    # file_num-is the number of file that contains the current sample to train
    # sample_num-is the number of current sample
    def add_train_data(self,file_num, sample_num, class_num):
        with open(self.path + '/data' + str(file_num) + '.txt', 'rb') as video:
            video = np.array(pickle.load(video))
        tmp = np.array(video[sample_num].matrix).reshape(1, 50, 85, 85).astype(np.int8)
        self.x_train.append(tmp)
        self.y_train.append(video[sample_num].label)
        self.cntr[class_num] = (self.cntr[class_num] + 1) % len(self.dictionary[class_num])
        if (self.cntr[class_num] == 0):
            self.flags[class_num] = False

    # because there's too big data for the RAM, the train shuld be modified half manually as below
    # as you can see each batch include 100 tain samples.
    #As wrote before dictionary[x][y] is an integer, the first to digits represents the file number, and the last 2 digits represent sample number in this file/
    def batch_train(self, batch_size):
        for i in range(batch_size/self.num_classes):
            for class_indx in self.num_classes:
                self.add_train_data(int(self.dictionary[class_indx][self.cntr[class_indx]] / 100),
                                                self.dictionary[class_indx][self.cntr[class_indx]] % 100, class_indx)
        self.y_train = np.array(self.y_train)
        self.x_train = np.array(self.x_train).astype(np.int8)
        self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=100)
        self.x_train=[]
        self.y_train=[]

    def train(self, batch_size):
        while (self.flags[0] or self.flags[1] or self.flags[2] or self.flags[3]):
            self.batch_train(batch_size)