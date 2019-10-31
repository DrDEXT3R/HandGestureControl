import cv2
import numpy as np

ESC = 27

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 127, 256, cv2.THRESH_BINARY)

    cv2.imshow("Camera", frame)
    cv2.imshow("Binary", binary_frame)

    k = cv2.waitKey(10)
    if k == ESC & 0xff:
        break

cap.release()
cv2.destroyAllWindows()