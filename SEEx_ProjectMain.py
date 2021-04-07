

import json
import platform
import os
from time import time, monotonic,sleep
import cv2
import numpy as np
import curses
import sys
import smbus2 as smbus
import codecs
import getch
import matplotlib.pylab as plt
import RPi.GPIO as GPIO # import the GPIO library
import SEEx_Function as SF

# SEEx Constants
global SEEx_time_0


I2C_SLAVE_ADDRESS = 0x0b # arduino slave address
I2Cbus = smbus.SMBus(1)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # use BCMf numbering for the pins on the Pi


# LED pins to signify blue or red balloon has been seen by Luxonis
LEDs = [4,17]
for i in LEDs:
    GPIO.setup(i, GPIO.OUT)
#    print(i)

# take time for SEEx robot (defined as global above)
SEEx_time_0 = time()



# Video Capture Object
cap = cv2.VideoCapture(0)

while (True):
    #capture frame-by-frame
    ret, frame = cap.read()
    
    # make a coppy
    frame_copy = np.copy(frame)
    
#     # Contour
#     left_screen, right_screen = SF.contourImage(frame_copy)
#     cv2.imshow('left', left_screen)
#     cv2.imshow('right', right_screen)
    
    
    # Lines
    height = frame_copy.shape[0]
    width = frame_copy.shape[1]
    
    # make a canny image
    canny_image = SF.canny(frame_copy)
    
    # regions of interest
    # must be an array of polygons
    roi_left  = np.array([
    [(0, height), (width, height), (int(width), 0), (0,0)]
    ])
    
    
    # left and right cropped images
    cropped_left = SF.region_of_interest(canny_image,roi_left)
    # lines on left and right
    left_lines = cv2.HoughLinesP(cropped_left, 100,np.pi/180, 10, np.array([]), minLineLength = 40, maxLineGap = 5)
#                     right_lines = cv2.HoughLinesP(cropped_right, 100,np.pi/180, 10, np.array([]), minLineLength = 10, maxLineGap = 5)
#     print(left_lines)

    # averaged lines
    ave_left_line_image = SF.average_slope_intercept(frame_copy, left_lines)
#     print(ave_left_line_image)
                        
    # Line images
    left_line_image = SF.display_lines(frame_copy, ave_left_line_image)
#     print(left_line_image)
#             
# 
#                         
    # Display Lines
    if left_line_image is not None:
        print("here too")
        left_combo_image = cv2.addWeighted(frame_copy, 0.5, left_line_image,1,1)
        cv2.imshow("LEFT", left_combo_image)
    else:
        cv2.imshow("LEFT", frame_copy)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()