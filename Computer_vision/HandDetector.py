import mediapipe as mp
import numpy as np

class Detector():
    
    def __init__(self,mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
               static_image_mode=mode,
               max_num_hands    =   maxHands,
               model_complexity =   1,
               min_detection_confidence =   detectionCon,
               min_tracking_confidence  =   trackCon)
        
        self.mpDraw = mp.solutions.drawing_utils
        
        self.points = np.empty((21,2))
        
    def findHands(self,frame,draw=True):
        
        h, w, c = frame.shape
        res = self.hands.process(frame)

        if res.multi_hand_landmarks:
            
            for hand in res.multi_hand_landmarks:

                if draw:
                    
                    self.mpDraw.draw_landmarks(frame, hand, self.mpHands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(hand.landmark):
              
              self.points[id,0] = int(lm.x*w)
              self.points[id,1] = int(lm.y*h)
        # else:
        #     self.points = np.empty_like(self.points)
        return frame




if __name__ == '__main__':

    import cv2
    
    detector = Detector()

    cap = cv2.VideoCapture(0)
    
    while 1:

        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        frame = detector.findHands(frame)



        cv2.imshow("Hand Detector", frame)
        if cv2.waitKey(1) == ord('q'):   # exit the program
                break
    