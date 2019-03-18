# -*- coding: utf-8 -*-

import imutils
import cv2
import numpy as np

interval = 30
outfilename = 'output.avi'
threshold=100.
fps = 10


cap = cv2.VideoCapture('2.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (480,640))
i=1
frames=[]
while(cap.isOpened()):
    ret, frame = cap.read()
    print(i)
    i=i+1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[40:395, 23:378]
    if ret==True:
        # write the flipped frame
        out.write(gray)
    frames.append(gray)
    flag =True
    if(flag):
        print(str(gray.ndim) + "\n"+str(gray.shape) + "\n" + str(gray.size))
        flag = False
    """if(flag):
        for j in gray.():
            for k in gray[0].length():
                print (gray[j][k] + " ")
        flag = False"""
   
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
         
cap.release()
cv2.destroyAllWindows()

