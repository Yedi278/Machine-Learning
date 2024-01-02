import cv2
import numpy as np
from time import time

import mediapipe as mp

# import pyautogui as pg


capture = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

hands = mpHands.Hands(False,1)
# h2, w2 = pg.size()

def closed(points,frame,h,w):
    
    avg = int(np.average(points[[5,9,13,17,0,1],0])),  int(np.average(points[[5,9,13,17,0,1],1]))
    avg2 = int(np.average(points[[4,8,12,16],0])),  int(np.average(points[[4,8,12,16],1]))

    cv2.circle(frame,avg,15,(255,0,0),cv2.FILLED)
    cv2.circle(frame,avg2,15,(255,0,0),cv2.FILLED)


    r = np.linalg.norm(points[0]-avg)

    cv2.circle(frame,avg,int(r),(255,0,0),4)

    for i in [4,8,12,16,20]:
        # cv2.circle(frame,(int(points[i,0]),int(points[i,1])),15,(255,0,255),cv2.FILLED)
        if np.linalg.norm(points[i]-avg) > r:
          return False
        
    return True



A = time()
while 1:


    ret, frame = capture.read()
    h, w, c = frame.shape
    frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    res = hands.process(frame_rgb)

    if res.multi_hand_landmarks:
          
          for hand in res.multi_hand_landmarks:
                
            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

            points = np.empty((21,2))
            for id, lm in enumerate(hand.landmark):
              
              points[id,0] = int(lm.x*w)
              points[id,1] = int(lm.y*h)
              
            # cv2.circle(frame,avg,15,(255,0,255),cv2.FILLED)
            
            if closed(points,frame,h,w): print("closed")
            else: print("open")


    
    B = time() - A
    fps = str(round(1/B))

    cv2.putText(frame,fps,(10,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,8,255))
    cv2.imshow("",frame)
    
    if cv2.waitKey(1) == ord('q'):   # exit the program
            break
    A = time()