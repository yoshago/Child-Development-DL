from src.Trainer import Trainer
from keras.layers import Dense, Flatten
from keras.layers.convolutional import (MaxPooling3D, Conv3D)


class CNN_Trainer(Trainer):

    def initial_model(self):
        # first layer
        self.model.add(Conv3D(10, (10, 10, 10), activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1),
                         input_shape=(1, 50, 85, 85)))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='same', name='pool1'))
        # 2nd layer group
        self.model.add(Conv3D(15, 5, 5, 5, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1)))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='same', name='pool2'))
        # 3rd layer group
        self.model.add(Conv3D(20, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1)))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='same', name='pool3'))
        self.model.add(Flatten())
        self.model.add(Dense(500, activation='sigmoid'))
        self.model.add(Dense(4, activation='softmax'))
        # compile model using accuracy to measure model performance
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self,name):
        self.model.save('../models/' + name +'.h5')