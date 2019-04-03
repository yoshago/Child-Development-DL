from keras.models import Sequential
from keras.layers import Dense
import pickle
import numpy as np

#create data for train
x_train=[]
y_train=[]
for j in range(1, 3):
    with open('../Data/data' + str(j) + '.txt', 'rb') as video:
        records_array=np.array(pickle.load(video))
        for i in records_array:
            tmp=np.array(i.matrix).reshape(340*340*150)
            x_train.append(tmp)
            y_train.append(i.label)
x_train = np.array(x_train)
y_train=np.array(y_train)
print(x_train.shape)
print(y_train.shape)

# create model
model = Sequential()
model.add(Dense(10, input_dim=150*340*340, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train,y_train, epochs=2, batch_size=1)

# evaluate the model
#scores = model.evaluate(x_train, y_train)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


x_train=[]
x_test=[]
with open('../Data/data' + str(86) + '.txt', 'rb') as test_video:
    test_records = np.array(pickle.load(test_video))
    for i in test_records:
        tmp = np.array(i.matrix).reshape(340 * 340 * 150)
        x_test.append(tmp)
x_test=np.array(x_test)
predictions = model.predict(x_test)
print(predictions)
