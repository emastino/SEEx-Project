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
###############################################################################################
###############################################################################################
#
# SEEx Functions
#
################################################################################################
def contourImage(frame):
    
    # Dimesnions
    height = frame.shape[0]
    width = frame.shape[1]
    
    # convert to HSV
    pp = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # color detection for YELLOW lines
    lower_yellow = np.array([15,75,20])
    upper_yellow = np.array([35,255,255])
    
    # yellow mask
    mask_yellow = cv2.inRange(pp,lower_yellow,upper_yellow)
    
    # left screen
    left_screen = mask_yellow[int(height/5):height, 0:int(width/2)]
    # right screen
    right_screen = mask_yellow[int(height/5):height, int(width/2):width]
    
    contours_left, hierarchy_left = cv2.findContours(left_screen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_right, hierarchy_right = cv2.findContours(right_screen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # initialize left contour area to 0
    left_contour_area = 0
    
    if len(contours_left) !=0:
        for contour in contours_left:
            if cv2.contourArea(contour)>300:
                left_contour_area = left_contour_area + cv2.contourArea(contour)
     
    # initialize right contour area to 0     
    right_contour_area = 0
    
    if len(contours_right) !=0:
        for contour in contours_right:
            if cv2.contourArea(contour)>300:
                right_contour_area = right_contour_area + cv2.contourArea(contour)
    
#     print(left_contour_area, right_contour_area)
#     # color detection for PURPLE lines
#     lower_purple = np.array([130,0,20])
#     upper_purple = np.array([160,255,255])
#     
#     
#     mask_purple = cv2.inRange(pp,lower_purple,upper_purple)
#     contours_purple, hierarchy = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     
#     if len(contours_purple) !=0:
#         for contour in contours_purple:
#             if cv2.contourArea(contour)>500:
#                 x,y,w,h = cv2.boundingRect(contour)
#                 cv2.rectangle(frame, (x,y), (x+w,y+h), (150,0,150), 3)
                    


#     driveCommands(left_contour_area, right_contour_area)
#     driveLeftLineCommands(left_contour_area, right_contour_area)
#     cv2.imshow("full screen", mask_yellow)

    return left_screen, right_screen
#     cv2.imshow("left", left_screen)
#     cv2.imshow("right", right_screen)
    
############################################################################################
def driveCommands(leftPix, rightPix):
    
    # total number of pixels
    totalPix = leftPix + rightPix
    
    # When too far left and we wish to turn right
    if leftPix >= rightPix + 200:
        percent = int(100*leftPix/totalPix)
        
        if percent >=100:
            commandTemp = "FR_" + str(percent)
        else:
            commandTemp = "FR_0" + str(percent)
        print(commandTemp)
    # Wehn too far right and we wish to turn left
    elif rightPix >= leftPix + 200:
        percent = int(100*rightPix/totalPix)
        if percent >=100:
            commandTemp = "FL_" + str(percent)
        else:
            commandTemp = "FL_0" + str(percent)
        
        print(commandTemp)
    # Go forward
    else:
        commandTemp = "FF_100"
        print(commandTemp)
        
    command = ConvertStringsToBytes(commandTemp)
    
    # Send command
#     sendMessage(command)
    
############################################################################################
def driveLeftLineCommands(leftPix, rightPix):
    print(leftPix, rightPix)
    # total number of pixels
    totalPix = leftPix + rightPix
    
    # threshold bound
    bound = 1000
    
    # arbitrary left number of pixels to not be over
    leftThreshold = 10000
    
    # arbitrary right number of pixels to not be over
    rightThreshold = 10000
    
    
    # When too far left and we wish to turn right
    if leftPix >= leftThreshold + bound and rightPix <= rightThreshold:
        
        print("CASE #1")
#         commandTemp = "FR_0" + str(50)
        # percent of leftPixels
        percent = int(100*leftThreshold/leftPix)
        
        if percent >=100:
            commandTemp = "FR_" + str(percent)
        else:
            commandTemp = "FR_0" + str(percent)
        print(commandTemp)
        
        
    # Wehn too far right and we wish to turn left
    elif leftPix <= leftThreshold - bound and rightPix <= rightThreshold:
#         commandTemp = "FF_100"
#         print(commandTemp)
        print("CASE #2")
#         commandTemp = "FL_0" + str(50)
        percent = int(100*leftPix/leftThreshold)
        
        if percent >=100:
            commandTemp = "FL_" + str(percent)
        elif percent == 0:
            commandTemp = "FL_" +str(50)
        else:
            commandTemp = "FL_0" + str(percent)
        
        print(commandTemp)
        
    # Run into a concave corner so we need to turn right
    elif leftPix >= leftThreshold + bound and rightPix >= rightThreshold:
        
        print("CASE #3")
#         commandTemp = "FR_0" + str(50)
        percent = int(100*rightThreshold/rightPix)
        
        if percent >=100:
            commandTemp = "FR_" + str(percent)
        else:
            commandTemp = "FR_0" + str(percent)
#         
        print(commandTemp)
        
        
    # Go forward otherwise
    else:
        commandTemp = "FF_100"
        print(commandTemp)
        
    command = ConvertStringsToBytes(commandTemp)
    
    # Send command
#     sendMessage(command)
    
############################################################################################

def canny(frame_passed):
    
    # make image gray scale
    gray = cv2.cvtColor(frame_passed,cv2.COLOR_RGB2GRAY)
    
    # Gaussian Blur the image
    blur = cv2.GaussianBlur(gray,(5,5),0) # 5x5 Kernel with deviation = 0
    
    # Make a canny image
    canny = cv2.Canny(blur,50, 255)

    return canny

############################################################################################

def region_of_interest(image, roi):
    # must be an array of polygons
    # roi_left  = np.array([
    # [(0, height/2), (0, height), (width/2, height), (width/2,height/2)]
    # ])
    # roi_right = np.array([
    # [(width/2, height/2), (width/2, height), (width, height), (width,height/2)]
    # ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,roi, 255)
    masked_image = cv2.bitwise_and(image,mask)
    
    return masked_image


############################################################################################

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    
    if lines is not None:
        
        for line in lines:
            print("Size of Line: ", np.size(line))
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)


############################################################################################

def average_slope_intercept(image, lines):
    
    angle_fit =[]
#     corner_fit =[]
  
    if lines is not None:
        
        for line in lines:
            
            x1, y1, x2, y2 = line.reshape(4)
            
            angle = np.arctan2((y2 -y1),(x2 -x1))
            angle_fit.append(angle)
#             print(x1,y1,x2,y2)
#             
#             if x1 == x2:
#                 break
#             else:
#                 parameters = np.polyfit((x1,x2), (y1,y2), 1) # gives back slope and y int for each line
#                 slope = parameters[0]
#                 intercept = parameters[1]
#             
#             # i think that corner lines will always have eith m >=0 and m <0
#             if slope < 0:
#                 line_fit.append((slope,intercept)) # left line 
#             else:
#                 corner_fit.append((slope, intercept)) # corners
#         
        # average slope and y ints of all the lines
        
        angle_fit_average = np.average(angle_fit, axis = 0)*180/np.pi
#         corner_fit_average = np.average(corner_fit, axis =0)
        print(angle_fit_average)
#         # line coordinates using he averages
#         left_line = make_coordinates(image, left_fit_average)
#         corner_line = make_coordinates(image, corner_fit_average)
    




############################################################################################

def make_coordinates(image, line_parameters):
      
    print("Line Parm", line_parameters, np.size(line_parameters))
    slope, intercept = line_parameters
    
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    
    
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    
    
    print(x1,y1,x2,y2)
    return np.array([x1,y1,x2,y2])
    
    
############################################################################################
        
def sendMessage(command):
    global SEEx_time_0
    
    # get current time
    current_time = time()
    # calculate time difference
    time_diff = current_time - SEEx_time_0
    
    if time_diff >= 0.1:  # if time difference greater than or equal to 0.1s, send commands
            I2Cbus.write_i2c_block_data(I2C_SLAVE_ADDRESS,0x00,command)
            SEEx_time_0 = time()
#     data = I2Cbus.read_i2c_block_data(I2C_SLAVE_ADDRESS,0x00,10)

############################################################################################
  
def ConvertStringsToBytes(src):
    converted = []
    for b in src:
        converted.append(ord(b))
    return converted


############################################################################################
###############################################################################################
################################################################################################
