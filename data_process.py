# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
from Read_csv import data, folder
from VidToMatrix import VidToMatrix

vid_counter=0
for i in range(len(data)):
    for file in os.listdir(folder+"\\"+str(data.Dir[i])):
        if file.endswith(".avi"):
            if "depth" not in file:
                all_data = (VidToMatrix(os.path.join(folder+"\\"+str(data.Dir[i]),file),data.Class[i]))
                vid_counter = vid_counter+1
                with open('Data\data' + str(vid_counter) + '.txt', 'wb') as fp:
                    pickle.dump(all_data, fp)
                
                
           





