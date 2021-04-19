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
import os
#from tensorflow import layers
import getch
import matplotlib.pylab as plt
import RPi.GPIO as GPIO # import the GPIO library
#import SEEx_Function as SF

# SEEx Constants
global SEEx_time_0

# I2C and GPIO set up
I2C_SLAVE_ADDRESS = 0x0b # arduino slave address
I2Cbus = smbus.SMBus(1)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # use BCMf numbering for the pins on the Pi

# Video Capture Object
cap = cv2.VideoCapture(0)

while (True):
    
    i = 0
    while i < 1500:
        #capture frame-by-frame
        ret, frame = cap.read()


        # make a coppy
        frame_copy = np.copy(frame)
        
        cv2.imwrite('picz\test'+str(i)+'.jpg',frame_copy)
        i += 1
        print(i)
        sleep(0.05)
    
cap.release()
cv2.destroyAllWindows()
