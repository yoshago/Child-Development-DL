# -*- coding: utf-8 -*-
import pickle
import cv2
 
records_array = []
for j in range(9,15):
    with open('Data/data' + str(j) + '.txt', 'rb') as fp:
        records_array.append(pickle.load(fp))
for i in records_array:
    cv2.imshow('frame',i[0][0][5])
    cv2.waitKey(1000)
