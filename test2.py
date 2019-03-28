# -*- coding: utf-8 -*-
import pickle
import cv2
 

cap = cv2.VideoCapture("2.avi")
ret = True
ret, frame = cap.read()
all_frames = []
i = 0
while(ret):
        i=i+1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[0:355, 0:355]
        cv2.imshow('frame', gray)
        cv2.waitKey(25)
        ret, frame = cap.read()  
        
        
