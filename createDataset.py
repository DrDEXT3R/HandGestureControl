import cv2
import numpy as np
import os


def nothing(x):
    pass


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


# Description of gestures
# 00 - nic
# 01 - otwarta dlon
# 02 - OK (palce razem)
# 03 - zacisnieta piesc
# 04 - ronaldinho
# 05 - pokoj


cv2.namedWindow('Settings')
cv2.createTrackbar('threshold', 'Settings', 50, 255, nothing)

backSub = cv2.createBackgroundSubtractorKNN()
# backSub = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)


    # Count the number of images of each gesture
    for i in range( len(os.listdir(dir)) ):
        img_sum[i] = len(os.listdir(dir + "/0" + str(i)))


    # Display the number of images of each gesture
    display_step = 10
    for i in reversed(range( len(os.listdir(dir)) )):
        cv2.putText(frame, "/0" + str(i) + ": "+ str(img_sum[i]), (10, frame.shape[0] - display_step), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
        display_step = display_step + 20
    cv2.putText(frame, "Counter", (10, frame.shape[0] - display_step - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)


    # ROI parameters
    bound_size = 2
    x1_roi = int(0.5*frame.shape[1]) - bound_size
    y1_roi = 10 - bound_size
    x2_roi = frame.shape[1]-10 + bound_size
    y2_roi = int(0.5*frame.shape[1]) + bound_size

    cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (0,255,0), bound_size)
    roi = frame[y1_roi+bound_size:y2_roi-bound_size, x1_roi+bound_size:x2_roi-bound_size]
 

    # ROI modification
    fgMask = backSub.apply(roi)
    threshold = cv2.getTrackbarPos('threshold', 'Settings')
    _, fgMask = cv2.threshold(fgMask, threshold, 255, cv2.THRESH_BINARY)
    fgMask = cv2.resize(fgMask, (128, 128)) 


    cv2.imshow("Frame", frame)
    cv2.imshow("Settings", fgMask)
    

    # Exit the program or save the image (ROI)
    k = cv2.waitKey(10)
    if k & 0xFF == ESC:
        break
    if k & 0xFF == ord('0'):
        cv2.imwrite(dir + '00/' + str(img_sum[0]) + '.jpg', fgMask)
    if k & 0xFF == ord('1'):
        cv2.imwrite(dir + '01/' + str(img_sum[1]) + '.jpg', fgMask)
    if k & 0xFF == ord('2'):
        cv2.imwrite(dir + '02/' + str(img_sum[2]) + '.jpg', fgMask)
    if k & 0xFF == ord('3'):
        cv2.imwrite(dir + '03/' + str(img_sum[3]) + '.jpg', fgMask)
    if k & 0xFF == ord('4'):
        cv2.imwrite(dir + '04/' + str(img_sum[4]) + '.jpg', fgMask)
    if k & 0xFF == ord('5'):
        cv2.imwrite(dir + '05/' + str(img_sum[5]) + '.jpg', fgMask)
    

cap.release()
cv2.destroyAllWindows()