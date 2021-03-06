# Description of gestures (directory - label name)
# 00 - none
# 01 - palm
# 02 - okay
# 03 - fist
# 04 - swing
# 05 - peace

import cv2
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import operator
from pynput.keyboard import Key, Controller

model = load_model('hand_gesture_recognition.h5')


def pretreatment(img):
    img = cv2.bilateralFilter(img,5,50,100)
    img = cv2.bilateralFilter(img,9,75,75)
    return img


def getROI(img, img_org):
    # ROI parameters
    bound_size = 2
    x1_roi = int(0.5*img.shape[1]) - bound_size
    y1_roi = 10 - bound_size
    x2_roi = img.shape[1]-10 + bound_size
    y2_roi = int(0.5*img.shape[1]) + bound_size

    cv2.rectangle(img_org, (x1_roi, y1_roi), (x2_roi, y2_roi), (0,255,0), bound_size)
    return img[y1_roi+bound_size:y2_roi-bound_size, x1_roi+bound_size:x2_roi-bound_size]


def bgSubtraction(img):
    mask = backSub.apply(img)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def morphologicalProcess(fgMask):
    kernel = np.ones((3,3),np.uint8)
    fgMask = cv2.dilate(fgMask,kernel,iterations = 2)

    kernel = np.ones((3,3),np.uint8)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5,5),np.uint8)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    return fgMask


def fillMask(fgMask):
    floodfilled_img = fgMask.copy()
    
    # Mask - flood filling (size +2 pixels)
    h, w = fgMask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    cv2.floodFill(floodfilled_img, mask, (0,0), 255);
    
    inverted = cv2.bitwise_not(floodfilled_img)
    
    # Get foreground (combine of two images)
    foreground = fgMask | inverted
    return foreground


def addTitle(img, title):
    title_bg = np.full((30,128), 255, dtype=np.uint8)
    cv2.putText(title_bg, title, (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
    img = np.concatenate((title_bg, img), axis=0)
    return img


ESC = 27

previousPrediction = 'none'
samePredictions = 1   # number of predictions in a row
reactionZone = 15 
control_ON = False
keyboard = Controller()

backSub = cv2.createBackgroundSubtractorKNN()

cap = cv2.VideoCapture(0)

while True:
    _, frame_org = cap.read()
    frame_org = cv2.flip(frame_org, 1)
    frame = pretreatment(frame_org)


    roi = getROI(frame, frame_org)

    fgMask = bgSubtraction(roi)

    fgMask = morphologicalProcess(fgMask)

    fgMask_filled = fillMask(fgMask)


    # Resize
    fgMask = cv2.resize(fgMask, (128, 128)) 
    fgMask_filled = cv2.resize(fgMask_filled, (128, 128))


    img_concatenation = np.concatenate((addTitle(fgMask, "Binary"), np.full((158,50), 255, dtype=np.uint8)), axis=1)

    img_concatenation = np.concatenate((img_concatenation, addTitle(fgMask_filled, "Filled")), axis=1)

    img_concatenation = np.concatenate((img_concatenation, np.full((100,306), 255, dtype=np.uint8)), axis=0)


    # Prediction
    result = model.predict(fgMask_filled.reshape(1, 128, 128, 1))

    prediction = {'none': result[0][0], 
                  'palm': result[0][1], 
                  'okay': result[0][2],
                  'fist': result[0][3],
                  'swing': result[0][4],
                  'peace': result[0][5]}

    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)


    cv2.putText(img_concatenation, "Detection: " + prediction[0][0], (5, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
    cv2.putText(frame_org, prediction[0][0], (int(0.5*frame.shape[1]), int(0.5*frame.shape[1]) + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)


    # Gesture control
    if prediction[0][0] != previousPrediction:
        samePredictions = 1
    else:
        if samePredictions > reactionZone:
            samePredictions = 0
        samePredictions = samePredictions + 1
    
    previousPrediction =  prediction[0][0]


    if samePredictions == reactionZone:

        # Toggle gesture control
        if prediction[0][0] == 'okay':
            control_ON = not control_ON
        
        # Gestures - VLC media player
        if control_ON == True:
            if prediction[0][0] == 'palm':      # Start / Pause
                keyboard.press(Key.space)
                keyboard.release(Key.space)
            elif prediction[0][0] == 'fist':    # Mute / Unmute 
                keyboard.press('m')
                keyboard.release('m')
            elif prediction[0][0] == 'swing':   # Volume down
                keyboard.press(Key.down)
                keyboard.release(Key.down)  
            elif prediction[0][0] == 'peace':   # Volume up
                keyboard.press(Key.up)
                keyboard.release(Key.up)

    
    cv2.putText(img_concatenation, "Gesture control: " + ('ON' if control_ON == True else 'OFF'), (5, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)

    cv2.imshow("Result", img_concatenation)
    cv2.imshow("Frame", frame_org)

    # Exit the program or save the image (ROI)
    k = cv2.waitKey(10)
    if k & 0xFF == ESC:
        break
    

cap.release()
cv2.destroyAllWindows()