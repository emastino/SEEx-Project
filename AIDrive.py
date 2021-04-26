import json
import platform
import os
from time import time, sleep
import cv2
import numpy as np
import curses
import sys
import smbus2 as smbus
import codecs
import tensorflow as tf
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

def driveCommands(predict, percent):
    print(predict, percent)
    
    command = ''
    
    if percent < 1:
        percent = '0'+str(int(percent * 100))
    else:
        percent = str(int(percent * 100))

    if predict == 'right':
        command = "FR_" + percent
    elif predict == 'left':
        command = "FL_" + percent
    else:
        command = "FF_" + percent
    print(command)    
    sendMessage(command)
      
    
############################################################################################
        
def sendMessage(command):
    senCom = ConvertStringsToBytes(command)
    try:
        I2Cbus.write_i2c_block_data(I2C_SLAVE_ADDRESS,0x00,senCom)
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

#check for rubber-banding
left = -1
right = 1
straight = 0
commandArray = [left,right,straight]
commandHist = []
num2send = 3

# Video Capture Object
cap = cv2.VideoCapture(0)

#Load the model
model = tf.keras.models.load_model('model2000.h5')

#define the model parameters
img_height = 256
img_width = 256
class_names = ['left', 'right', 'straight']

while (True):
    
    # get current time
    current_time = time()

    # calculate time difference
    time_diff = current_time - SEEx_time_0
    
    if time_diff >= 0.65:  # if time difference greater than or equal to 0.1s, send commands
        #capture frame-by-frame
        ret, frame = cap.read()    
    
        # make a coppy
        frame_copy = np.copy(frame)
    
        #make prediction
        src = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (img_height, img_width)).astype("float32")
        img = np.expand_dims(src, axis=0)

        predictions = model.predict(img)
        score = tf.nn.softmax(predictions[0])
        predict = class_names[np.argmax(score)]
        percent = np.max(score)
        
        newCommand = commandArray[np.argmax(score)]

        # SEND COMMANDS
        #check to make sure we arent rubber-banding
        if len(commandHist) == 4:
            evens = sum(np.array(commandHist)[[0,2]])
            odds = sum(np.array(commandHist)[[1,3]])
            onetwo = sum(np.array(commandHist)[[0,1]])
            threefour = sum(np.array(commandHist)[[2,3]])
            
            if (abs(odds) == 2 and abs(evens) == 2) or (abs(onetwo) == 2 and abs(threefour) == 2):
                if odds == -evens and abs(odds) > 1 and newCommand == commandHist[1]:
                    command2send = class_names[np.argmax(np.array(commandArray) == -newCommand)]
                    for i in range(num2send):
                        driveCommands(command2send, 1)
                        commandHist.append(-newCommand)
                        old = commandHist.pop()
                elif onetwo == -threefour and abs(onetwo) > 1 and newCommand == commandHist[2]:
                    command2send = class_names[np.argmax(np.array(commandArray) == -newCommand)]
                    for i in range(num2send):
                        driveCommands(command2send, 1)
                        commandHist.append(-newCommand)
                        old = commandHist.pop()
                else:
                    driveCommands(predict, percent)
                    commandHist.append(newCommand)
                    old = commandHist.pop()
            else:
                driveCommands(predict, percent)
                commandHist.append(newCommand)
                old = commandHist.pop()
        else:
            driveCommands(predict, percent)
            commandHist.append(newCommand)
            
        SEEx_time_0 = time()
    else:
        continue
    #cv2.imshow("changed brightness", frame_copy)
    #cv2.imshow("mask", tots)
#     sleep(0.085)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
