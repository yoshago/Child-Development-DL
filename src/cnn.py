import pickle
import tensorflow as tf
from keras.layers import Dense, Flatten
import numpy as np
from keras.layers.convolutional import (MaxPooling3D, Conv3D)
from keras.models import Sequential
from random import shuffle, random

# each of thus data keys storage locations of data of class 1/2/3/4
data_key_class1=[]
data_key_class2=[]
data_key_class3=[]
data_key_class4=[]
# we use thus flags and counters to stop when we done pass throw all the data
flag1=True
flag2=True
flag3=True
flag4=True
cntr_class1=0
cntr_class2=0
cntr_class3=0
cntr_class4=0
# matrix for train and label data
x_train=np.array([])
y_train=np.array([])
# CNN MODEL
model = Sequential()

def initial_model():
    #first layer
    model.add(Conv3D(10, (10, 10, 10), activation='relu', border_mode='same', name='conv1',subsample=(1, 1, 1), input_shape=(1,50,85,85)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='same', name='pool1'))
    #2nd layer group
    model.add(Conv3D(15, 5, 5, 5, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='same', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(20, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='same', name='pool3'))
    model.add(Flatten())
    model.add(Dense(500, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))
    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

import os
print(os.getcwd())

#load data key
def load_dictionary():
    with open('Data/compressed_data/data_key_new.txt', 'rb') as data_key:
        data_key = np.array(pickle.load(data_key))
    print(data_key[0])
    data_key_class1=data_key[0]
    data_key_class2=data_key[1]
    data_key_class3=data_key[2]
    data_key_class4=data_key[3]
    shuffle(data_key_class1)
    shuffle(data_key_class2)
    shuffle(data_key_class3)
    shuffle(data_key_class4)
#thre is 95 files of data, each of them divided for samples. the data_key array (from above) contains locations and classifications).
#file_num-is the number of file that contains the current sample to train
#sample_num-is the number of current sample
def add_train_data(file_num,sample_num,cntr_class, flag):
    with open('Data/compressed_data/data' + str(file_num) + '.txt', 'rb') as video:
        video = np.array(pickle.load(video))
    tmp = np.array(video[sample_num].matrix).reshape(1, 50, 85, 85).astype(np.int8)
    x_train.append(tmp)
    y_train.append(video[sample_num].label)
    cntr_class = (cntr_class + 1) % len(data_key_class1)
    if (cntr_class == 0):
        flag = False
    return cntr_class, flag

# def randomal_train(rand):
#     if (rand > 0.8 and rand < 0.85):
#         for i in range(25):
#             cntr_class1 = add_train_data(int(data_key_class1[cntr_class1]) / 100, data_key_class1[cntr_class1] % 100, cntr_class1, flag1)
#         cntr_class1 = (cntr_class1 - 25) % len(data_key_class1)
#     if (rand > 0.85 and rand < 0.9):
#         for i in range(25):
#             cntr_class2 = add_train_data(int(data_key_class2[cntr_class2]) / 100, data_key_class2[cntr_class2] % 100, cntr_class2, flag1)
#         cntr_class2 = (cntr_class2 - 25) % len(data_key_class2)
#     if (rand > 0.9 and rand < 0.95):
#         for i in range(25):
#             cntr_class3 = add_train_data(int(data_key_class3[cntr_class3]) / 100, data_key_class3[cntr_class3] % 100, cntr_class3, flag3)
#         cntr_class3 = (cntr_class3 - 25) % len(data_key_class3)
#     if (rand > 0.95 and rand < 1.0):
#         for i in range(25):
#             cntr_class4 = add_train_data(int(data_key_class4[cntr_class4]) / 100, data_key_class4[cntr_class4] % 100, cntr_class4, flag1)
#         cntr_class4 = (cntr_class4 - 25) % len(data_key_class4)
#     else:
#         return
#     y_train = np.array(y_train)
#     x_train = np.array(x_train)
#     model.fit(x_train, y_train, epochs=5, batch_size=25)


# because there's too big data for the RAM, the train shuld be modified half manually as below
# as you can see each batch include 100 tain samples(we added chance to train the model on one type of data, to make him out from local minimum
def batch_train():
    for i in range(25):
        cntr_class1, flag1 = add_train_data(int(data_key_class1[cntr_class1] / 100),data_key_class1[cntr_class1] % 100, cntr_class1, flag1)
        cntr_class2, flag2 = add_train_data(int(data_key_class2[cntr_class2] / 100),data_key_class2[cntr_class2] % 100, cntr_class2, flag2)
        cntr_class3, flag3 = add_train_data(int(data_key_class3[cntr_class3] / 100),data_key_class3[cntr_class3] % 100, cntr_class3, flag3)
        cntr_class3, flag4 = add_train_data(int(data_key_class4[cntr_class4] / 100),data_key_class4[cntr_class4] % 100, cntr_class4, flag4)
    y_train = np.array(y_train)
    x_train = np.array(x_train).astype(np.int8)
    model.fit(x_train, y_train, epochs=10, batch_size=100)
    x_train = []
    y_train = []

'''
x_test=[]
for j in range(1, 95):
    with open('Data/compressed_data/data' + str(j) + '.txt', 'rb') as video:
        test_records = np.array(pickle.load(video))
        if (int(test_records[0].name)== 4 or int(test_records[0].name) == 5 or int(test_records[0].name) == 19 or int(test_records[0].name) == 11):
            for i in test_records:
                tmp = np.array(i.matrix).reshape(50*85*85)
                x_test.append(tmp)
            x_test=np.array(x_test)
            predictions = model.predict(x_test)
            print("suppose to be:" + str(test_records[0].label))
            print(predictions)
            x_test = []
'''
def main():
    load_dictionary()
    initial_model()
    while (flag1 or flag2 or flag3 or flag4):
        batch_train()

main()