import pickle
import numpy as np
import skimage
from src.Feature import Feature

NUM_OF_VIDEOS=94 # num of videos in our database
## This function compress the data because it's too big for our RAM
## take as input a record
##for each chunck in this video the function drop 2 of 3 frames (take only when %3=0)
##and take the max pixel from each 4x4 square in this frame.
def compress_rec(rec):
    new_rec=[]
    classification=rec[0].label
    for data in rec:
        new_mat=[]
        for k in range(len(data.matrix)):
            if (k%3==0):
                new_frame=np.array(data.matrix[k])
                new_frame=skimage.measure.block_reduce(new_frame, (4, 4), np.max)
                new_mat.append(new_frame)
        new_rec.append(Feature(new_mat,rec[0].name,rec[0].label))
        new_mat=np.array(new_mat)
        print(new_mat.shape)
    return new_rec

## func data_key() make .txt file that represent map for all the data
## As you can see below i used constant 100.
## I use this const (like in hotel room number, each video file represent 'floor' and each video have divide to small chunks-'rooms')  to give number for each data sample.
##for an example in data key the number 2313 mean video 23 chunk 13
def data_key():
    data_key=[[],[],[],[]]
    for i in range(1,NUM_OF_VIDEOS+1):
        with open('../Data/compressed_data/data' + str(i) + '.txt', 'rb') as data:
            print(i)
            records=np.array(pickle.load(data))
            for j in range(len(records)):
                data_key[records[j].label.index(1)].append(100*i+j)
    print(data_key[0])
    print(data_key[1])
    print(data_key[2])
    print(data_key[3])
    print(len(data_key[0]))
    print(len(data_key[1]))
    print(len(data_key[2]))
    print(len(data_key[3]))
    with open('../Data/compressed_data/data_key_new.txt', 'wb') as data:
        pickle.dump(np.array(data_key), data)

'''
for j in range(1, NUM_OF_VIDEOS+1):
    with open('../Data/data' + str(j) + '.txt', 'rb') as video:
        records_array=np.array(pickle.load(video))
        rec=compress_rec(records_array)
        with open('../Data/compressed_data/data' + str(j) + '.txt', 'wb') as fp:
            pickle.dump(rec, fp)
        #print("rec label: " + str(rec[0].label) + "          " + str(j))
        #print("rec name:  " + rec[0].name + "                " + str(j))
        #print("original record shape:  "  + str(np.array(records_array[2].matrix).shape))
'''
data_key()

