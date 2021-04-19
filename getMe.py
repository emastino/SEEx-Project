import json
import platform
import os
from time import time,sleep
import cv2
import numpy as np
import curses
import sys
import smbus2 as smbus
import codecs
import getch
import matplotlib.pylab as plt
# import RPi.GPIO as GPIO # import the GPIO library
import subprocess



# SEEx Constants
global SEEx_time_0

# I2C and GPIO set up
# I2C_SLAVE_ADDRESS = 0x0b # arduino slave address
# I2Cbus = smbus.SMBus(1)
# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BCM)  # use BCMf numbering for the pins on the Pi


# # LED pins to signify blue or red balloon has been seen by Luxonis
# LEDs = [4,17]
# for i in LEDs:
#     GPIO.setup(i, GPIO.OUT)
# #    print(i)

# take time for SEEx robot (defined as global above)
SEEx_time_0 = time()
delayTime = 0.2
###############################################################################################
###############################################################################################
#
# SEEx Functions
#
############################################################################################
def driveCommands(input):
    if (input == 'w'):
        commandTemp = "FF_100"
        command = ConvertStringsToBytes(commandTemp)
    elif (input == 'a'):
        commandTemp = "FL_100"
        command = ConvertStringsToBytes(commandTemp)
    elif (input == 's'):
        commandTemp = "BB_100"
        command = ConvertStringsToBytes(commandTemp)
    elif (input == 'd'):
        commandTemp = "FF_100"
        command = ConvertStringsToBytes(commandTemp)
    else:
        commandTemp = "xx_000"
        command = ConvertStringsToBytes(commandTemp)
        
    print(commandTemp)
    sendMessage(command)
        
############################################################################################
        
def sendMessage(command):
    
    
    global SEEx_time_0
    
    # get current time
    current_time = time()

    # calculate time difference
    time_diff = current_time - SEEx_time_0
    
    # Force message
    FM = 1
    
    
    if time_diff >= delayTime:  # if time difference greater than or equal to 0.1s, send commands
        
       while FM == 1: 
            try:
                I2Cbus.write_i2c_block_data(I2C_SLAVE_ADDRESS,0x00,command)
                SEEx_time_0 = time()
                FM=0
            except:
                FM=1
                print("Remote I/O ERROR")
                

############################################################################################
  
def ConvertStringsToBytes(src):
    converted = []
    for b in src:
        converted.append(ord(b))
    return converted


############################################################################################
###############################################################################################
################################################################################################

while (True):
    
    cmd = getch.getche()
    driveCommands(cmd)