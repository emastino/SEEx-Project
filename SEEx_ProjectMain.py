

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
    lower_yellow = np.array([20,50,20])
    upper_yellow = np.array([40,255,255])
    
    # yellow mask
    mask_yellow = cv2.inRange(pp,lower_yellow,upper_yellow)
    
    # right screen
#     right_screen = mask_yellow[int(0):height, int(width/2):width]
    
    contours_left, hierarchy_left = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours_right, hierarchy_right = cv2.findContours(right_screen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    
    # initialize left contour area to 0
#     left_contour_area = 0
#     
#     if len(contours_left) !=0:
#         for contour in contours_left:
#             if cv2.contourArea(contour)>300:
#                 left_contour_area = left_contour_area + cv2.contourArea(contour)
#                 
                
#                 (x,y,w,h) = cv2.boundingRect(contour)
# #               
#                 
#                 rect = cv2.minAreaRect(contour)
#                 box = cv2.boxPoints(rect)
#                 box = np. int0(box)
#                 
#                 return box
#                 cv2.drawContours(frame, [box], 0, (0,191,255),2)
     
    # initialize right contour area to 0     
#     right_contour_area = 0
#     
#     if len(contours_right) !=0:
#         for contour in contours_right:
#             if cv2.contourArea(contour)>300:
#                 right_contour_area = right_contour_area + cv2.contourArea(contour)
    
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
    return mask_yellow

###########################################################################################

def contourImageROICounter(frame_passed):
    
    frame = np.copy(frame_passed)
    
    # Dimesnions
    height = frame.shape[0]
    width = frame.shape[1]
    contour_area = np.sum(frame==255)
    
# #     # convert to HSV
# #     pp = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
# 
#     # color detection for YELLOW lines
#     lower_white = np.array([0,0,180])
#     upper_white = np.array([255,255,255])
#     
#     # yellow mask
#     mask = cv2.inRange(frame,lower_white,upper_white)
# 
#     
#     contours, hierarchy_left = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     
#     # initialize left contour area to 0
#     contour_area = 0
#     
#     if len(contours) !=0:
#         for contour in contours:
#             if cv2.contourArea(contour)>10000:
#                 contour_area = contour_area + cv2.contourArea(contour)
                
    
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
def driveCommands(leftPix, rightPix):
    print(leftPix, rightPix)
    # total number of pixels
    totalPix = leftPix + rightPix
    if totalPix >0:
        dummyPercent = int(100*leftPix/totalPix)
    else:
        dummyPercent =0
        
    # When too far left and we wish to turn right
    if leftPix >= rightPix + 1000:
        percent = int(100*leftPix/totalPix)
        
        if percent >=100:
            commandTemp = "FR_" + str(percent)
            command = ConvertStringsToBytes(commandTemp)
            sendMessage(command)
            print(commandTemp)
            
        elif percent <= 52.5 and percent >= 47.5:
            commandTemp = "FR_100"
            command = ConvertStringsToBytes(commandTemp)
            turnTimes = np.random.randint(5) + 10
            print(turnTimes)
            for i in range(1,3):
                sendMessage(command)
                sleep(0.1)
                print("TORN! GO RIGHT")
                
        else:
            commandTemp = "FR_0" + str(percent)
#             commandTemp = "FR_100"
            command = ConvertStringsToBytes(commandTemp)
            sendMessage(command)
            print(commandTemp)
        
        
    # Wehn too far right and we wish to turn left
    elif rightPix >= leftPix + 1000:
        
        percent = int(100*rightPix/totalPix)
        
        if percent >=100:
            commandTemp = "FL_" + str(percent)
            command = ConvertStringsToBytes(commandTemp)
            sendMessage(command)
            print(commandTemp)
            
        elif percent <= 52.5 and percent >= 47.5:
            
            commandTemp = "FL_100"
            command = ConvertStringsToBytes(commandTemp)
            turnTimes = np.random.randint(5) + 5
            print(turnTimes)
            for i in range(1, 3):
                sendMessage(command)
                sleep(0.1)
                print("TORN! GO LEFT")  
        
        else:
            commandTemp = "FL_0" + str(percent)
#             commandTemp = "FL_100"
            command = ConvertStringsToBytes(commandTemp)
            sendMessage(command)
            print(commandTemp)
                
    elif dummyPercent <=51 and dummyPercent > 49:
        commandTemp = "FL_100"
        command = ConvertStringsToBytes(commandTemp)
        for i in range(1, 12):
            sendMessage(command)
            sleep(0.1)
            print("DUMMY 180")
        
    # Go forward
    else:
        commandTemp = "FF_100"
        command = ConvertStringsToBytes(commandTemp)       
        # Send command
        sendMessage(command)
        print("JUST CRUZIN'")
        print(commandTemp)
    
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
    sendMessage(command)
    
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
    global SEEx_time_0
    
    # get current time
    current_time = time()

    # calculate time difference
    time_diff = current_time - SEEx_time_0
    
    if time_diff >= 0.1:  # if time difference greater than or equal to 0.1s, send commands
        try:
            I2Cbus.write_i2c_block_data(I2C_SLAVE_ADDRESS,0x00,command)
            SEEx_time_0 = time()
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



# Video Capture Object
cap = cv2.VideoCapture(0)

while (True):
    #capture frame-by-frame
    ret, frame = cap.read()
    
    
    # make a coppy
    frame_copy = np.copy(frame)
    
#     frame_copy = SF.adjust_brightness(frame_copy_temp, 75)
    
#     # Contour
#     left_screen, right_screen = SF.contourImage(frame_copy)
#     cv2.imshow('left', left_screen)
#     cv2.imshow('right', right_screen)
    
    
    masked_yellow = contourImage(frame_copy)
#     if left_box is not None:
#         cv2.drawContours(frame_copy, [left_box], 0, (0,191,255),2)
  
    
    # Lines
    height = frame_copy.shape[0]
    width = frame_copy.shape[1]
#     print(height,width)
#     # make a canny image
#     canny_image = canny(frame_copy)
#     
#     cv2.imshow("CANNY", canny_image)
#     # regions of interest
#     # must be an array of polygons
    roi_left_V  = np.array([
    [(0, height), (int(width/2), height), (int(width/4), 0), (0,0)]
    ])
    left_V_mask = region_of_interest(masked_yellow, roi_left_V)
    
    number_on_left_V = contourImageROICounter(left_V_mask)
    
    
    
    
    roi_right_V  = np.array([
    [(int(width/2), height), (width, height), (width,0),(int(3*width/4), 0)]
    ])
    right_V_mask = region_of_interest(masked_yellow, roi_right_V)
    
    number_on_right_V = contourImageROICounter(right_V_mask)
    print(number_on_left_V, number_on_right_V)
    
    tots = right_V_mask + left_V_mask
    
    cv2.imshow("mask", tots)
    # left and right cropped images
#     cropped_left = SF.region_of_interest(canny_image,roi_left)
    # lines on left and right
#     left_lines = cv2.HoughLinesP(cropped_left, 100,np.pi/180, 10, np.array([]), minLineLength = 10, maxLineGap = 1)
#                     right_lines = cv2.HoughLinesP(cropped_right, 100,np.pi/180, 10, np.array([]), minLineLength = 10, maxLineGap = 5)
#     print(left_lines)
    # Display Lines
#     if left_lines is not None:
#         print("here too")
#         all_lines_image = SF.display_lines(frame_copy, left_lines)
#         left_combo_image_l = cv2.addWeighted(frame_copy, 0.9, all_lines_image,1,1)
#         cv2.imshow("ALL LINES", left_combo_image_l)
# 
#     else:
#         cv2.imshow("ALL LINES", frame_copy)

#     # averaged lines
#     ave_left_line_image = SF.average_slope_intercept(frame_copy, left_lines)
# #     print(ave_left_line_image)
#                         
#     # Line images
#     left_line_image = SF.display_lines(frame_copy, ave_left_line_image)
#     print(left_line_image)
#                                    
#     # Display Lines
#     if left_line_image is not None:
# #         print("here too")
#         left_combo_image = cv2.addWeighted(frame_copy, 0.9, left_line_image,1,1)
#         cv2.imshow("LEFT", left_combo_image)
#     else:
#         cv2.imshow("LEFT", frame_copy)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()