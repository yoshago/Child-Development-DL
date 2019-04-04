import pickle
import numpy as np
import skimage
from src.Feature import Feature

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







for j in range(1, 95):
    with open('../Data/data' + str(j) + '.txt', 'rb') as video:
        records_array=np.array(pickle.load(video))
        rec=compress_rec(records_array)
        with open('../Data/compressed_data/data' + str(j) + '.txt', 'wb') as fp:
            pickle.dump(rec, fp)
        #print("rec label: " + str(rec[0].label) + "          " + str(j))
        #print("rec name:  " + rec[0].name + "                " + str(j))
        #print("original record shape:  "  + str(np.array(records_array[2].matrix).shape))




