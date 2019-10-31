import cv2
import numpy as np

ESC = 27
cap = cv2.VideoCapture(0)

#backSub = cv2.createBackgroundSubtractorMOG2()
#backSub = cv2.createBackgroundSubtractorMOG2(0, 100)
backSub = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    fgMask = backSub.apply(frame)
    


    #fgMask = backSub.apply(frame, learningRate=0)
    #kernel = np.ones((3, 3), np.uint8)
    #fgMask = cv2.erode(fgMask, kernel, iterations=1)
    #res = cv2.bitwise_and(frame, frame, mask=fgMask)


    
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    


    _, binary = cv2.threshold(fgMask, 127, 255, cv2.THRESH_BINARY)
    

    #fgMask = cv2.bilateralFilter(fgMask, 5, 50, 100) 
    


    cv2.imshow('FG Mask', fgMask)

    cv2.imshow('Res', binary)


    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    cv2.imshow("Camera", frame)

    
    k = cv2.waitKey(10)
    if k == ESC & 0xff:
        break

cap.release()
cv2.destroyAllWindows()