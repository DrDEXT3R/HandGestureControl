import cv2
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import operator

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


ESC = 27


# Description of gestures
# 00 - nic
# 01 - otwarta dlon
# 02 - OK (palce razem)
# 03 - zacisnieta piesc
# 04 - ronaldinho
# 05 - pokoj


backSub = cv2.createBackgroundSubtractorKNN()
# backSub = cv2.createBackgroundSubtractorMOG2()

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


    cv2.imshow("Frame", frame_org)
    img_concatenation = np.concatenate((fgMask, fgMask_filled), axis=1)
    cv2.imshow("Preview: binary & filled", img_concatenation) 


    # Prediction
    result = model.predict(fgMask_filled.reshape(1, 128, 128, 1))

    prediction = {'00': result[0][0], 
                  '01': result[0][1], 
                  '02': result[0][2],
                  '03': result[0][3],
                  '04': result[0][4],
                  '05': result[0][5]}


    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    print('Result: ' + prediction[0][0])




    # Exit the program or save the image (ROI)
    k = cv2.waitKey(10)
    if k & 0xFF == ESC:
        break
    

cap.release()
cv2.destroyAllWindows()