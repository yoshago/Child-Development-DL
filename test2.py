# -*- coding: utf-8 -*-
import pickle
import cv2
 
records_array = []
for j in range(46,51):
    with open('Data/data' + str(j) + '.txt', 'rb') as fp:
        records_array.append(pickle.load(fp))
for i in records_array:

    for x in range (18):
        cv2.imshow('frame',i[x][0][70])
        cv2.waitKey(3000)
print (i[-1][0][70].shape +"11 23 29 46 47 48 49 50")
