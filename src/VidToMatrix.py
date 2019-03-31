import cv2
import numpy as np
from src.Feature import Feature

def VidToMatrix(file, label, format, name):
    cap = cv2.VideoCapture(file)
    frames=[]
    ret = True
    ret, frame = cap.read()
    all_frames = []
    frame_counter = 0
    while(ret):
        frame_counter=frame_counter+1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if format==1:
            gray = gray[40:380, 23:363]
        elif format==2:
            gray = gray[0:340, 0:340]
            gray = np.pad(gray,((0,0),(0,4)),'constant')
        elif format==3:
            gray = gray[0:340, 20:360]
        elif format==4:
            gray = gray[0:340, 23:363]                     
        
        
        frames.append(gray)
        if frame_counter%150==0:
            all_frames.append(Feature(frames,name,label))
            frames = []
        ret, frame = cap.read()      
        

    cap.release()
    print(len(all_frames))
    return all_frames