
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
from tensorflow import layers
import getch
import matplotlib.pylab as plt
import RPi.GPIO as GPIO # import the GPIO library
import SEEx_Function as SF
import pdb

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
    
    if percent < 100:
        percent = '0'+str(int(percent * 100))
    else:
        percent = str(int(percent * 100))

    if predict == 'right':
        command = "FR_" + percent
    elif predict == 'left':
        command = "FL_" + percent
    else:
        command = "FF_" + percent
        
    sendMessage(command)
      
    
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

pdb.set_trace()
#Load the model
model = tf.keras.models.load_model('init_model.h5')

#define the model parameters
img_height = 256
img_width = 256
class_names = ['left', 'right', 'straight']

while (True):
    #capture frame-by-frame
    ret, frame = cap.read()
    
    
    # make a coppy
    frame_copy = np.copy(frame)
    
    
    
    #make prediction
    img = tf.keras.preprocessing.image.smart_resize(frame_copy, size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
       
    # SEND COMMANDS
    driveCommands(prediction, score)
    
#     cv2.imshow("changed brightness", frame_copy)
#     cv2.imshow("mask", tots)
    sleep(0.085)


    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
