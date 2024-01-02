import cv2
import numpy as np
import mediapipe as mp
from HandDetector import Detector as hd

detector = hd()

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()

    frame = detector.findHands(frame)

    array = detector.points

    avg = int(np.average(array[:,0])), int(np.average(array[:,1]))
    cv2.circle(frame,avg,15,(255,0,0),cv2.FILLED)
    

    cv2.imshow("Hand Detector",frame)
    if cv2.waitKey(1) == ord('q'):   # exit the program
        break
    