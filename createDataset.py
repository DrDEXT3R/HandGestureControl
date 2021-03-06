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
dir = "dataset/"


# Create directories
if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir  + "/00")
    os.makedirs(dir  + "/01")
    os.makedirs(dir  + "/02")
    os.makedirs(dir  + "/03")
    os.makedirs(dir  + "/04")
    os.makedirs(dir  + "/05")


img_sum = [None] * len(os.listdir(dir))

backSub = cv2.createBackgroundSubtractorKNN()

cap = cv2.VideoCapture(0)

while True:
    _, frame_org = cap.read()
    frame_org = cv2.flip(frame_org, 1)
    frame = pretreatment(frame_org)


    # Count the number of images of each gesture
    for i in range( len(os.listdir(dir)) ):
        img_sum[i] = len(os.listdir(dir + "/0" + str(i)))


    # Display the number of images of each gesture
    display_step = 10
    for i in reversed(range( len(os.listdir(dir)) )):
        cv2.putText(frame_org, "/0" + str(i) + ": "+ str(img_sum[i]), (10, frame_org.shape[0] - display_step), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
        display_step = display_step + 20
    cv2.putText(frame_org, "Counter", (10, frame_org.shape[0] - display_step - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)


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


    # Exit the program or save the image (ROI)
    k = cv2.waitKey(10)
    if k & 0xFF == ESC:
        break
    if k & 0xFF == ord('0'):
        cv2.imwrite(dir + '00/' + str(img_sum[0]) + '.jpg', fgMask_filled)
    if k & 0xFF == ord('1'):
        cv2.imwrite(dir + '01/' + str(img_sum[1]) + '.jpg', fgMask_filled)
    if k & 0xFF == ord('2'):
        cv2.imwrite(dir + '02/' + str(img_sum[2]) + '.jpg', fgMask_filled)
    if k & 0xFF == ord('3'):
        cv2.imwrite(dir + '03/' + str(img_sum[3]) + '.jpg', fgMask_filled)
    if k & 0xFF == ord('4'):
        cv2.imwrite(dir + '04/' + str(img_sum[4]) + '.jpg', fgMask_filled)
    if k & 0xFF == ord('5'):
        cv2.imwrite(dir + '05/' + str(img_sum[5]) + '.jpg', fgMask_filled)
    

cap.release()
cv2.destroyAllWindows()