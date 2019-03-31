
# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
from Read_csv import data, folder
from VidToMatrix import VidToMatrix
"""
exporting data to .txt file in two ways:
    1. each video to .txt file
    2. each video divided to small video in .txt video
"""
vid_counter=0
for i in range(len(data)):
    for file in os.listdir(folder+"\\"+str(data.Dir[i])):
        if file.endswith(".avi"):
            if "depth" not in file:
                classification = [0,0,0,0]
                classification[data.Class[i]-1]=1
                name = str(data.Dir[i])
                if name[-1]=='a' or name[-1]=='b':
                    name = name[:-1]
                all_data = VidToMatrix(os.path.join(folder+"\\"+str(data.Dir[i]),file),classification, int(data.format[i]),name)
                vid_counter = vid_counter+1
                with open('Data\data' + str(vid_counter) + '.txt', 'wb') as fp:
                    pickle.dump(all_data, fp)
                
           





