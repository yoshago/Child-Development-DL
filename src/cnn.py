
import pickle
import tensorflow as tf

import numpy as np
from keras.layers.convolutional import (MaxPooling3D, Conv3D)
from keras.models import Sequential


#define test
with open('Data/data' + str(16) + '.txt', 'rb') as test_video:
    test_record_array = pickle.load(test_video)
    x_test=np.array([test_record_array[i][0]for i in range(len(test_record_array)-1)])
    y_test=np.array([test_record_array[i][1]for i in range(len(test_record_array)-1)])

# train the model
x_train=[]
for j in range(1, 15):
    with open('Data/data' + str(j) + '.txt', 'rb') as video:
        records_array=np.array(pickle.load(video))
        print('the ' + str(j)+ ' video, shape: '+str(records_array.shape))
        for i in range(len(records_array) - 1):
            x_train.append(records_array[i][0])
            #print(x_train.shape)
            y_train=np.array([records_array[i][1]for i in range(len(records_array)-1)])
    x_train = np.array(x_train)
    x_train.reshape(-1,len(records_array) - 1,150,355,355)
    y_train.reshape(-1, 1)

# There are four array:

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential()

#first layer
model.add(Conv3D(64, (3, 3, 3), activation='relu',border_mode='same', name='conv1',subsample=(1, 1, 1), input_shape=(150,355,355,1)))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1'))
#2nd layer group
model.add(Conv3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2'))
# 3rd layer group
model.add(Conv3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1)))
model.add(Conv3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3'))



# compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)

#        model.predict(i[0])
#       print('ACTUAL VALUE IS: ' +str(i[1])+'\n')

""",validation_data=(x_test, y_test), epochs=3,shuffle=True"""
"""
## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

## add max pooling to obtain the most imformatic features
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

## perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer2 = BatchNormalization()(pooling_layer2)
flatten_layer = Flatten()(pooling_layer2)

## create an MLP architecture with dense layers : 4096 -> 512 -> 10
## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=10, activation='softmax')(dense_layer2)

## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)
"""