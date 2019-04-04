from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import pickle
import numpy as np


# create model
model = Sequential()
model.add(Dense(200, input_dim=50*85*85, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
        model.fit(x_train, y_train, epochs=1, batch_size=10)
    x_train=[]
    y_train=[]

# evaluate the model
#scores = model.evaluate(x_train, y_train)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


x_train=[]
x_test=[]
for j in range(1, 95):
    with open('../Data/compressed_data/data' + str(j) + '.txt', 'rb') as video:
        test_records = np.array(pickle.load(video))
        if (int(test_records[0].name)== 4 or int(test_records[0].name) == 5):
            for i in test_records:
                tmp = np.array(i.matrix).reshape(50*85*85)
                x_test.append(tmp)
            x_test=np.array(x_test)
            predictions = model.predict(x_test)
            print("suppose to be:" + str(test_records[0].label) )
            print(predictions)
            x_test = []