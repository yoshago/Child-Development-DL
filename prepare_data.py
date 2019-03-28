# -*- coding: utf-8 -*-
import pickle
import cv2
"""
dividing each file to smaller file of X frames.
X defined in vidToMatrix.
"""
def prepare():
    cntr_array=[0,0,0,0]
    records_array = []
    ## load each video to records_array
    for j in range (1,82):
        print('Dividing video:  '+'Data/data' + str(j) + '.txt' )
        with open('Data/data' + str(j) + '.txt', 'rb') as video:
            records_array=pickle.load(video)
            print(' lable: ' + str(records_array[0][1]))
    ## divide each video to num of frames (defined in vidToMatrix) and export to Data/data/<lable number>/<number of video>.txt
            for i in range (len(records_array)-1):
                with open('Data/divide_data/' + str(records_array[i][1])+'/'+ str(cntr_array[records_array[i][1]-1]) + '.txt', 'wb') as sub_vid:
                    pickle.dump(records_array[i], sub_vid)
                cntr_array[records_array[i][1]-1]=cntr_array[records_array[i][1]-1]+1
                print('    '+str(i)+ ' from ' + str(len(records_array)-2))
            print('num of data videos:  ' + str(len(records_array)) + '\n')
#   print(str(len(records_array[i][0])) + '\n' + str(records_array[i][1]) + '\n\n')
#        cv2.imshow('frame',i[-1][0][x])
#       cv2.waitKey(25)
#  cv2.waitKey(3000)
#    print(i[-1][0][25].shape)

