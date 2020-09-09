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

def find_contour_with_max_area(contours):
    max_area = -1
    for i in range(len(contours)):
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > max_area:
            max_area = area
            index = i
            
    return max_area, index

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
    frame_copy = frame.copy()
    cv2.rectangle(frame, (10,10), (450,450), (255, 0, 0), 2)
    result = remove_background(frame)
    roi = result[10:450, 10:450]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9, 9), 0)
    threshold = cv2.getTrackbarPos("Threshold", "Settings")
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Capture", frame)
    cv2.imshow("Result", thresh)
    
    # Find contours
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        _, index = find_contour_with_max_area(contours)
        max_area = contours[index]
        hull = cv2.convexHull(max_area)
        cv2.drawContours(frame_copy, [max_area], 0, (0, 255, 0), 2)
        cv2.drawContours(frame_copy, [hull], 0, (0, 0, 255), 3)
    
    cv2.imshow("Output", frame_copy)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
