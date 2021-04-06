

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
    
    # Contour
    left_screen, right_screen = SF.contourImage(frame_copy)
    
    # Display Left Screen
    cv2.imshow('left', left_screen)
    
    # Display RGB
    cv2.imshow('rgb', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()