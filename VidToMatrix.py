import cv2
import numpy as np

def VidToMatrix(file, label, flag):
    cap = cv2.VideoCapture(file)
    frames=[]
    ret = True
    ret, frame = cap.read()
    all_frames = []
    i = 0
    while(ret):
        i=i+1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[40:395, 23:378]
        if not flag:
            gray = gray[0:355, 0:355]
        frames.append(gray)
        if i%150==0:
            all_frames.append([frames, label])
            frames= []
        ret, frame = cap.read()      
        
    if len(frames)>0:
        all_frames.append([frames, label])
    cap.release()
    print(len(all_frames))
    return all_frames