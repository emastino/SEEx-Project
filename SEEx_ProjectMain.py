

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
import RPi.GPIO as GPIO # import the GPIO library
import subprocess



# SEEx Constants
global SEEx_time_0

# I2C and GPIO set up
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
delayTime = 0.075

# Stuck Logic
global currentCommand ,previousCommand, stuckCounter
currentCommand = "FF"
previousCommand = "FF"

stuckCounter = 0
###############################################################################################
###############################################################################################
#
# SEEx Functions
#
################################################################################################

def adjust_brightness(image, level):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b = np.mean(image[:,:,2])
    
    if b == 0:
        return image
    
    r = level/b
    c = image.copy()
    
    c[:,:,2] = c[:,:,2]*r
    
    return cv2.cvtColor(c, cv2.COLOR_HSV2BGR)


################################################################################################

def contourImage(frame_passed):
    frame = np.copy(frame_passed)
    # Dimesnions
    height = frame.shape[0]
    width = frame.shape[1]
    
    # convert to HSV
    pp = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    
    # color detection for YELLOW lines
#     lower_yellow = np.array([0,0,245])
#     upper_yellow = np.array([150,255,255])
    lower_yellow = np.array([25,10,0])
    upper_yellow = np.array([35,255,255])
    # yellow mask
    mask_yellow = cv2.inRange(pp,lower_yellow,upper_yellow)
    
    # right screen
#     right_screen = mask_yellow[int(0):height, int(width/2):width]
    
    contours_left, hierarchy_left = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours_right, hierarchy_right = cv2.findContours(right_screen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    kernel = np.ones((5,5), np.uint8)
    out_image = cv2.morphologyEx( mask_yellow, cv2.MORPH_OPEN,kernel)
    return out_image

###########################################################################################

def contourImageROICounter(frame_passed):
    
    frame = np.copy(frame_passed)
    
    # Dimesnions
    height = frame.shape[0]
    width = frame.shape[1]
    contour_area = np.sum(frame==255)

    return contour_area
############################################################################################
 
def order_box(box):
    
    srt = np.argsort(box[:,1])
    btm1 = box[srt[0]]
    btm2 = box[srt[1]]
    
    top1 = box[srt[2]]
    top2 = box[srt[3]]
    
    bc = btm1[0] < btm2[0]
    btm_l = btm1 if bc else btm2
    btm_r = btm2 if bc else btm1
    
    tc = top1[0] < top2[0]
    top_l = top1 if tc else top2
    top_r = top2 if tc else top1
    
    return np.array([top_l, top_r, btm_r, btm_l])

    
############################################################################################
def driveCommands_2(LV, RV, BL, BR):
    global currentCommand, previousCommand, stuckCounter
    
    # total number of pixels on V
    total_V = LV+RV
    
    total_B = BL + BR
    
    
    if total_B >0:
        dummyPercent = int(100*BL/total_B)
    else:
        dummyPercent =0
    
    
    
    if total_B > 15000:
        
        print("CASE #1")
        BL_percent = dummyPercent
        BR_percent = 100 - dummyPercent
        
        if BL_percent > BR_percent + 10:
         
            # BACK UP
            commandTemp = "BB_075"
            command = ConvertStringsToBytes(commandTemp)

            for i in range(2):
                sendMessage(command)
                sleep(delayTime)
                print("GO BACK")
            
            # Turn to right
            commandTemp = "FR_075"
            command = ConvertStringsToBytes(commandTemp)
            for i in range(3):
                sendMessage(command)
                sleep(delayTime)

           
           
           
        elif BR_percent > BL_percent + 10:
            
            # BACK UP
            commandTemp = "BB_075"
            command = ConvertStringsToBytes(commandTemp)
        
            for i in range(2):
                sendMessage(command)
                sleep(delayTime)
                print("TORN!")
                
                
            #turn to left
            commandTemp = "FR_075"
            command = ConvertStringsToBytes(commandTemp)
            for i in range(3):
                sendMessage(command)
                sleep(delayTime)
                print("TORN! GO LEFT 90") 
            
            
            
        else:
            commandTemp = "BB_075"
            command = ConvertStringsToBytes(commandTemp)

            for i in range(2):
                sendMessage(command)
                sleep(delayTime)
                print("TORN! GO RIGHT 90")
            # turn 180 to left
            commandTemp = "FR_075"
            command = ConvertStringsToBytes(commandTemp)
            for i in range(7):
                sendMessage(command)
                sleep(delayTime)
                print("TURN LEFT 180")
                
            currentCommand = "  "
            
    else:
        
        
        print("CASE #2")
        # When too far left and we wish to turn right
        if LV >= RV + 8000:
            percent = int(100*LV/total_V)
            
            if percent >=100:
                commandTemp = "FR_" + str(percent)
                command = ConvertStringsToBytes(commandTemp)
                 #turn to left
                for i in range(2):
                    sendMessage(command)
                    sleep(delayTime)
                    print(commandTemp)
                
                    
            else:
                commandTemp = "FR_0" + str(percent)
    #             commandTemp = "FR_100"
                command = ConvertStringsToBytes(commandTemp)
                for i in range(2):
                    sendMessage(command)
                    sleep(delayTime)
                    print(commandTemp)
            
            currentCommand = "FR"
        # Wehn too far right and we wish to turn left
        elif RV >= LV + 8000:
            
            percent = int(100*RV/total_V)
            
            if percent >=100:
                commandTemp = "FL_" + str(percent)
                command = ConvertStringsToBytes(commandTemp)
                for i in range(2):
                    sendMessage(command)
                    sleep(delayTime)
                    print(commandTemp)
            
            else:
                commandTemp = "FL_0" + str(percent)
    #             commandTemp = "FL_100"
                command = ConvertStringsToBytes(commandTemp)
                sendMessage(command)
                for i in range(2):
                    sendMessage(command)
                    sleep(delayTime)
                    print(commandTemp)
                    
            currentCommand = "FL" 
        else:
            commandTemp = "FF_100"
            command = ConvertStringsToBytes(commandTemp)       
            # Send command
            sendMessage(command)
            print("JUST CRUZIN'")
            print(commandTemp)
            
            currentCommand = "FF"
        
            
        # STUCK LOGIC
        if currentCommand == previousCommand:
            stuckCounter+=1
            print("STUCK COUNTER: ", stuckCounter)
            previousCommand = currentCommand
        else:
            stuckCounter = 0
                
############################################################################################

def canny(frame_passed):
    
    # make image gray scale
    gray = cv2.cvtColor(frame_passed,cv2.COLOR_RGB2GRAY)
    
    # Gaussian Blur the image
    blur = cv2.GaussianBlur(gray,(5,5),0) # 5x5 Kernel with deviation = 0
    
    # Make a canny image
    canny = cv2.Canny(blur,100, 150)

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
            x1,y1,x2,y2 = line.reshape(4)
            print(x1,y1,x2,y2)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image


############################################################################################

def average_slope_intercept(image, lines):
    
#     angle_fit =[]
#     line_fit =[]
    if lines is not None:
        average_coord = np.average(lines, axis =0)
#         print(average_coord)
        left_line = make_coordinates(image, average_coord)
        return np.array([left_line])
    else:
        return np.array([[0,0,0,0]])
#     print(average_coord)
#     if lines is not None:
#         
#         for line in lines:
#             
#             x1, y1, x2, y2 = line.reshape(4)
#             
# #             angle = np.arctan2((y2 -y1),(x2 -x1))
# #             angle_fit.append(angle)
# #             print(x1,y1,x2,y2)
#             
#             if x1 == x2:
#                 slope = 10000
#             else:
#                 parameters = np.polyfit((x1,x2), (y1,y2), 1) # gives back slope and y int for each line
#                 slope = parameters[0]
#                 intercept = parameters[1]
#             
#             line_fit.append((slope,intercept)) # left line 
#             
# #         
#         # average slope and y ints of all the lines
#         
# #         angle_fit_average = np.average(angle_fit, axis = 0)*180/np.pi
#         left_fit_average = np.average(line_fit,axis=0)
    
#         print(angle_fit_average)
#         # line coordinates using he averages
#         left_line = make_coordinates(image, left_fit_average)
#         corner_line = make_coordinates(image, corner_fit_average)
    




############################################################################################

def make_coordinates(image, coords):
      
    height = image.shape[0]
    width = image.shape[1]
    
    x1, y1, x2, y2 = coords.reshape(4)
    
    # calculate slope
    if x1 == x2:
        x_1 = int(x1)
        y_1 = int(height)
        x_2 = int(x1)
        y_2 = 0
    else:
        m = (y2-y1)/(x2-x1)
            
        # calculate y intercept
        b = int(y1 - m*x1)
        
        # Check the intercepts
        
        # if m is negative
        if m <= 0: 
            if b > height:
                # intersect bottome of screen so use y = height and then solve for x
                y_1 = int(height)
                x_1 = int((y_1 - b)/m) 
            else:
                # intersecting left of screen
                x_1 = 0
                y_1 = int(b)
                
            y_right = int(m*width + b)
            
            if y_right >= 0:
                # intersect the right side of the wall
                x_2 = int(width)
                y_2 = int(m*x_2 + b)
                
            else:
                # intersect top wall
                y_2 = 0
                x_2 = int(-b/m)
            
            
        else:
            if b >= 0:
                y_1 = int(b)
                x_1 = 0
            else:
                y_1 = 0
                x_1 = int((y_1 -b)/m)
                
            y_right = int(m*width + b)
            
            if y_right >= height:
                
                y_2 = int(height)
                x_2 = int((y_2 -b)/m)
                
            else:
                
                x_2 = int(width)
                y_2 = int(m*x_2 + b)
            

#     print(x1,y1,x2,y2)
    return np.array([x_1,y_1,x_2,y_2])
    
    
############################################################################################
        
def sendMessage(command):
    try:
        I2Cbus.write_i2c_block_data(I2C_SLAVE_ADDRESS,0x00,command)
        
    except:
        print("Remote I/O ERROR")
            
        
#     data = I2Cbus.read_i2c_block_data(I2C_SLAVE_ADDRESS,0x00,10)
#         I2Cbus.read_i2c_block_data(I2C_SLAVE_ADDRESS,0x00,10)

############################################################################################
  
def ConvertStringsToBytes(src):
    converted = []
    for b in src:
        converted.append(ord(b))
    return converted


############################################################################################
###############################################################################################
################################################################################################

# Camera properties
# cam_props = {'brightness': 10, 'contrast':0, 'saturation': 5,
#             'sharpness': 70, 'exposure_auto': 1, 'exposure_absolute':25,
#              'focus_auto': 0, 'focus_absolute': 30,
#              'white_balance_temperature_auto':0,'white_balance_temperature':3300}
# 
# for key in cam_props:
#     
#     subprocess.call(['v4l2-ctl -d /dev/video0 -c {}={}'.format(key, str(cam_props[key]))], shell=True)


# Video Capture Object
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,0)

sleep(1)

while (True):
    
    #capture frame-by-frame
    ret, frame = cap.read()
    # get current time
    current_time = time()

    # calculate time difference
    time_diff = current_time - SEEx_time_0
    
    
    if time_diff >= delayTime:
        
        # make a coppy
        frame_copy = np.copy(frame)
        
    #     frame_copy = SF.adjust_brightness(frame_copy_temp, 10)

        
        # get contours of yellow line
        contours_yellow = contourImage(frame_copy)
    #     if left_box is not None:
    #         cv2.drawContours(frame_copy, [left_box], 0, (0,191,255),2)
      
        
        # Lines
        height = frame_copy.shape[0]
        width = frame_copy.shape[1]
        
        # Left Contour
        roi_left_V  = np.array([
        [(int(0), int(7*height/10)), (int(width/2), int(7*height/10)), (int(2*width/5), int(height/10)), (int(width/10),int(height/10))]
        ])
        
        left_V_mask = region_of_interest(contours_yellow, roi_left_V)
        
        number_on_left_V = contourImageROICounter(left_V_mask)
        
        # Right Contour
        roi_right_V  = np.array([
            [(int(width/2), int(7*height/10)), (int(width), int(7*height/10)), (int(9*width/10),int(height/10)),(int(3*width/5), int(height/10))]
            ])
        
        right_V_mask = region_of_interest(contours_yellow, roi_right_V)
        
        number_on_right_V = contourImageROICounter(right_V_mask)
            
            
        # Contours for bottom left and right boxes
        
        roi_BL  = np.array([
        [(0, int(7*height/10)), (int(width/2), int(7*height/10)), (int(width/2), height), (0,height)]
        ])
        
        BL_mask = region_of_interest(contours_yellow, roi_BL)
        
        number_on_BL = contourImageROICounter(BL_mask)
        
        # Right Contour
        roi_BR  = np.array([
        [(int(width/2), int(7*height/10)), (int(width), int(7*height/10)), (int(width), height), (int(width/2),height)]
        ])
        
        BR_mask = region_of_interest(contours_yellow, roi_BR)
        
        number_on_BR = contourImageROICounter(BR_mask)
        
        
        # Pixels on V_L, V_R, B_L, and B_R
        print(number_on_left_V, number_on_right_V, number_on_BL, number_on_BR)
        
        tots = right_V_mask + left_V_mask + BL_mask + BR_mask
        
        
        # SEND COMMANDS
        driveCommands_2(number_on_left_V , number_on_right_V , number_on_BL, number_on_BR)
        
        SEEx_time_0 = time()
    else:
        continue
    
    # Do this if 75 of the same commands have been sent
    if stuckCounter >= 100:
        # BACK UP
        commandTemp = "BB_075"
        command = ConvertStringsToBytes(commandTemp)
        
        for i in range(7):
            sendMessage(command)
            sleep(delayTime)
            print("YOU'RE STUCK")
            
        stuckCounter = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
