# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:58:17 2020

@author: PJ
"""

import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)

def nothing(x):
    pass

def remove_background(frame):
    fgmask = bgModel.apply(frame,learningRate=0)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    result = cv2.bitwise_and(frame, frame, mask=fgmask)
    return result

def create_settings_window():
    cv2.namedWindow("Settings")
    cv2.resizeWindow("Settings", 700, 350)
    cv2.createTrackbar("Threshold", "Settings", 0, 255, nothing)
    cv2.setTrackbarPos("Threshold", "Settings", 0)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720) 
create_settings_window()

bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (10,10), (450,450), (255, 0, 0), 2)
    result = remove_background(frame)
    roi = result[10:450, 10:450]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9, 9),0)
    #cv2.imshow("Blur", blur)
    threshold = cv2.getTrackbarPos("Threshold", "Settings")
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Capture", frame)
    cv2.imshow("Result", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
