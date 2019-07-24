from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pickle
import numpy as np
from random import shuffle


# create model
model = Sequential()
model.add(Dense(300, input_dim=50*85*85, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(4, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#load data key
with open('../Data/compressed_data/data_key.txt', 'rb') as data_key:
    data_key = np.array(pickle.load(data_key))
    print("len of all data is:  " + str(len(data_key)))
shuffle(data_key)
cntr=0
x_train=[]
y_train=[]
while(cntr<len(data_key)):
    for i in range(100):
        if(i+cntr<len(data_key)):
            with open('../Data/compressed_data/data' + str(int(data_key[i+cntr]/100)) + '.txt', 'rb') as video:
                video=np.array(pickle.load(video))
                tmp = np.array(video[data_key[i+cntr]%100].matrix).reshape(50 * 85 * 85).astype(np.int8)
                x_train.append(tmp)
                y_train.append(video[data_key[i+cntr]%100].label)

    y_train = np.array(y_train).astype(np.int8)
    x_train = np.array(x_train).astype(np.int8)
    model.fit(x_train, y_train, epochs=15, batch_size=20)
    x_train = []
    y_train = []
    cntr = cntr + 100
'''
#create data for train
x_train=[]
y_train=[]
for num in range(19):
    for j in range(1+5*num, (num+1)*5 ):
        with open('../Data/compressed_data/data' + str(j) + '.txt', 'rb') as video:
            records_array=np.array(pickle.load(video))
            if(int(records_array[0].name)!=4 and int(records_array[0].name)!=5):
                for i in records_array:
                    tmp=np.array(i.matrix).reshape(50*85*85)
                    x_train.append(tmp)
                    y_train.append(i.label)
    y_train=np.array(y_train)
    x_train = np.array(x_train)
    # Fit the model
    if(x_train.size!=0):
        model.fit(x_train, y_train, epochs=1, batch_size=15)
    x_train=[]
    y_train=[]
'''
# evaluate the model
#scores = model.evaluate(x_train, y_train)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


x_test=[]
for j in range(1, 95):
    with open('../Data/compressed_data/data' + str(j) + '.txt', 'rb') as video:
        test_records = np.array(pickle.load(video))
        if (int(test_records[0].name)== 4 or int(test_records[0].name) == 5 or int(test_records[0].name) == 19 or int(test_records[0].name) == 11):
            for i in test_records:
                tmp = np.array(i.matrix).reshape(50*85*85)
                x_test.append(tmp)
            x_test=np.array(x_test)
            predictions = model.predict(x_test)
            print("suppose to be:" + str(test_records[0].label) )
            print(predictions)
            x_test = []