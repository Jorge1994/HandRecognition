# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:58:17 2020

@author: PJ
"""

import cv2
import numpy as np
import math

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

def find_fingers(frame, max_contour):
    hull = cv2.convexHull(max_contour, returnPoints = False)
    defects = cv2.convexityDefects(max_contour, hull)
    if defects is None:
        defects = [0]
        num_defects = 0
    else:
        num_defects = 0
        for i in range (defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2)/(2 * b * c))
            semiperimenter = (a + b + c) / 2
            herao_area = math.sqrt(semiperimenter * (semiperimenter - a) * (semiperimenter - b) * (semiperimenter - c))
            height = (herao_area * 2) / a
            
            if angle <= math.pi / 2 and height > 30:
                num_defects += 1
                cv2.circle(frame, far, 3, (255,0,0), -1)
                
            cv2.line(frame, start, end, (0,0,255), 2)
                
        
    return defects, num_defects

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
        max_contour = contours[index]
        hull = cv2.convexHull(max_contour)
        cv2.drawContours(frame_copy, [max_contour], 0, (0, 255, 0), 2)
        cv2.drawContours(frame_copy, [hull], 0, (0, 0, 255), 3)
        find_fingers(frame_copy, max_contour)
    
    cv2.imshow("Output", frame_copy)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
