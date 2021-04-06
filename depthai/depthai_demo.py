#!/usr/bin/env python3

# To Run: ./depthai_demo.py -cnn tiny-yolo-v3 -sh 4 -nce 1



import json
import platform
import os
from time import time, monotonic,sleep
import cv2
import numpy as np
import depthai
from depthai_helpers.version_check import check_depthai_version
from depthai_helpers.object_tracker_handler import show_tracklets
from depthai_helpers.config_manager import DepthConfigManager
from depthai_helpers.arg_manager import CliArgs


###############################################################################################
###############################################################################################
# Libs and Packages imported for SEEx project
import curses
import sys
import smbus2 as smbus
import codecs
import getch
import matplotlib.pylab as plt
import RPi.GPIO as GPIO # import the GPIO library

# import time # created conflict with depthai code
###############################################################################################
###############################################################################################

print('Using depthai module from: ', depthai.__file__)
print('Depthai version installed: ', depthai.__version__)
if platform.machine() not in ['armv6l', 'aarch64']:
    check_depthai_version()

is_rpi = platform.machine().startswith('arm') or platform.machine().startswith('aarch64')
global args, cnn_model2, SEEx_time_0



###############################################################################################
###############################################################################################
# SEEx Constants

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
def contourImage(frame_passed):
    
    # make copy of image
    frame = np.copy(frame_passed)
    
    # Dimesnions
    height = frame.shape[0]
    width = frame.shape[1]
    
    # convert to HSV
    pp = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # color detection for YELLOW lines
    lower_yellow = np.array([25,75,20])
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
    cv2.imshow("left", left_screen)
    cv2.imshow("right", right_screen)
    
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


class DepthAI:
    global is_rpi
    process_watchdog_timeout = 10  # seconds
    nnet_packets = None
    data_packets = None
    runThread = True

    def reset_process_wd(self):
        global wd_cutoff
        wd_cutoff = monotonic() + self.process_watchdog_timeout
        return

    def on_trackbar_change(self, value):
        self.device.send_disparity_confidence_threshold(value)
        return

    def stopLoop(self):
        self.runThread = False

    def startLoop(self):
        cliArgs = CliArgs()
        args = vars(cliArgs.parse_args())

        configMan = DepthConfigManager(args)
        if is_rpi and args['pointcloud']:
            raise NotImplementedError("Point cloud visualization is currently not supported on RPI")
        # these are largely for debug and dev.
        cmd_file, debug_mode = configMan.getCommandFile()
        usb2_mode = configMan.getUsb2Mode()

        # decode_nn and show_nn are functions that are dependent on the neural network that's being run.
        decode_nn = configMan.decode_nn
        show_nn = configMan.show_nn

        # Labels for the current neural network. They are parsed from the blob config file.
        labels = configMan.labels
        NN_json = configMan.NN_config

        # This json file is sent to DepthAI. It communicates what options you'd like to enable and what model you'd like to run.
        config = configMan.jsonConfig

        # Create a list of enabled streams ()
        stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in configMan.stream_list]

        enable_object_tracker = 'object_tracker' in stream_names

        # grab video file, if option exists
        video_file = configMan.video_file

        self.device = None
        if debug_mode:
            print('Cmd file: ', cmd_file, ' args["device_id"]: ', args['device_id'])
            self.device = depthai.Device(cmd_file, args['device_id'])
        else:
            self.device = depthai.Device(args['device_id'], usb2_mode)

        print(stream_names)
        print('Available streams: ' + str(self.device.get_available_streams()))

        # create the pipeline, here is the first connection with the device
        p = self.device.create_pipeline(config=config)

        if p is None:
            print('Pipeline is not created.')
            exit(3)

        nn2depth = self.device.get_nn_to_depth_bbox_mapping()

        t_start = time()
        frame_count = {}
        frame_count_prev = {}
        nnet_prev = {}
        nnet_prev["entries_prev"] = {}
        nnet_prev["nnet_source"] = {}
        frame_count['nn'] = {}
        frame_count_prev['nn'] = {}

        NN_cams = {'rgb', 'left', 'right'}

        for cam in NN_cams:
            nnet_prev["entries_prev"][cam] = None
            nnet_prev["nnet_source"][cam] = None
            frame_count['nn'][cam] = 0
            frame_count_prev['nn'][cam] = 0

        stream_windows = []
        for s in stream_names:
            if s == 'previewout':
                for cam in NN_cams:
                    stream_windows.append(s + '-' + cam)
            else:
                stream_windows.append(s)

        for w in stream_windows:
            frame_count[w] = 0
            frame_count_prev[w] = 0

        tracklets = None

        self.reset_process_wd()

        time_start = time()

        def print_packet_info_header():
            print('[hostTimestamp streamName] devTstamp seq camSrc width height Bpp')

        def print_packet_info(packet, stream_name):
            meta = packet.getMetadata()
            print("[{:.6f} {:15s}]".format(time() - time_start, stream_name), end='')
            if meta is not None:
                source = meta.getCameraName()
                if stream_name.startswith('disparity') or stream_name.startswith('depth'):
                    source += '(rectif)'
                print(" {:.6f}".format(meta.getTimestamp()), meta.getSequenceNum(), source, end='')
                print('', meta.getFrameWidth(), meta.getFrameHeight(), meta.getFrameBytesPP(), end='')
            print()
            return

        def keypress_handler(self, key, stream_names):
            cams = ['rgb', 'mono']
            self.cam_idx = getattr(self, 'cam_idx', 0)  # default: 'rgb'
            cam = cams[self.cam_idx]
            cam_c = depthai.CameraControl.CamId.RGB
            cam_l = depthai.CameraControl.CamId.LEFT
            cam_r = depthai.CameraControl.CamId.RIGHT
            cmd_ae_region = depthai.CameraControl.Command.AE_REGION
            cmd_exp_comp  = depthai.CameraControl.Command.EXPOSURE_COMPENSATION
            cmd_set_focus = depthai.CameraControl.Command.MOVE_LENS
            cmd_set_exp   = depthai.CameraControl.Command.AE_MANUAL
            keypress_handler_lut = {
                ord('f'): lambda: self.device.request_af_trigger(),
                ord('1'): lambda: self.device.request_af_mode(depthai.AutofocusMode.AF_MODE_AUTO),
                ord('2'): lambda: self.device.request_af_mode(depthai.AutofocusMode.AF_MODE_CONTINUOUS_VIDEO),
                # 5,6,7,8,9,0: short example for using ISP 3A controls for Mono cameras
                ord('5'): lambda: self.device.send_camera_control(cam_l, cmd_ae_region, '0 0 200 200 1'),
                ord('6'): lambda: self.device.send_camera_control(cam_l, cmd_ae_region, '1000 0 200 200 1'),
                ord('7'): lambda: self.device.send_camera_control(cam_l, cmd_exp_comp, '-2'),
                ord('8'): lambda: self.device.send_camera_control(cam_l, cmd_exp_comp, '+2'),
                ord('9'): lambda: self.device.send_camera_control(cam_r, cmd_exp_comp, '-2'),
                ord('0'): lambda: self.device.send_camera_control(cam_r, cmd_exp_comp, '+2'),
            }
            if key in keypress_handler_lut:
                keypress_handler_lut[key]()
            elif key == ord('c'):
                if 'jpegout' in stream_names:
                    self.device.request_jpeg()
                else:
                    print("'jpegout' stream not enabled. Try settings -s jpegout to enable it")
            elif key == ord('s'):  # switch selected camera for manual exposure control
                self.cam_idx = (self.cam_idx + 1) % len(cams)
                print("======================= Current camera to control:", cams[self.cam_idx])
            # RGB manual focus/exposure controls:
            # Control:      key[dec/inc]  min..max
            # exposure time:     i   o    1..33333 [us]
            # sensitivity iso:   k   l    100..1600
            # focus:             ,   .    0..255 [far..near]
            elif key == ord('i') or key == ord('o') or key == ord('k') or key == ord('l'):
                max_exp_us = int(1000*1000 / config['camera'][cam]['fps'])
                self.rgb_exp = getattr(self, 'rgb_exp', 20000)  # initial
                self.rgb_iso = getattr(self, 'rgb_iso', 800)  # initial
                rgb_iso_step = 50
                rgb_exp_step = max_exp_us // 20  # split in 20 steps
                if key == ord('i'): self.rgb_exp -= rgb_exp_step
                if key == ord('o'): self.rgb_exp += rgb_exp_step
                if key == ord('k'): self.rgb_iso -= rgb_iso_step
                if key == ord('l'): self.rgb_iso += rgb_iso_step
                if self.rgb_exp < 1:     self.rgb_exp = 1
                if self.rgb_exp > max_exp_us: self.rgb_exp = max_exp_us
                if self.rgb_iso < 100:   self.rgb_iso = 100
                if self.rgb_iso > 1600:  self.rgb_iso = 1600
                print("===================================", cam, "set exposure:", self.rgb_exp, "iso:", self.rgb_iso)
                exp_arg = str(self.rgb_exp) + ' ' + str(self.rgb_iso) + ' 33333'
                if cam == 'rgb':
                    self.device.send_camera_control(cam_c, cmd_set_exp, exp_arg)
                elif cam == 'mono':
                    self.device.send_camera_control(cam_l, cmd_set_exp, exp_arg)
                    self.device.send_camera_control(cam_r, cmd_set_exp, exp_arg)
            elif key == ord(',') or key == ord('.'):
                self.rgb_manual_focus = getattr(self, 'rgb_manual_focus', 200)  # initial
                rgb_focus_step = 3
                if key == ord(','): self.rgb_manual_focus -= rgb_focus_step
                if key == ord('.'): self.rgb_manual_focus += rgb_focus_step
                if self.rgb_manual_focus < 0:   self.rgb_manual_focus = 0
                if self.rgb_manual_focus > 255: self.rgb_manual_focus = 255
                print("========================================== RGB set focus:", self.rgb_manual_focus)
                focus_arg = str(self.rgb_manual_focus)
                self.device.send_camera_control(cam_c, cmd_set_focus, focus_arg)
            return

        for stream in stream_names:
            if stream in ["disparity", "disparity_color", "depth"]:
                cv2.namedWindow(stream)
                trackbar_name = 'Disparity confidence'
                conf_thr_slider_min = 0
                conf_thr_slider_max = 255
                cv2.createTrackbar(trackbar_name, stream, conf_thr_slider_min, conf_thr_slider_max, self.on_trackbar_change)
                cv2.setTrackbarPos(trackbar_name, stream, args['disparity_confidence_threshold'])

        right_rectified = None
        pcl_converter = None

        ops = 0
        prevTime = time()
        if args['verbose']: print_packet_info_header()
        while self.runThread:
            # retreive data from the device
            # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
            self.nnet_packets, self.data_packets = p.get_available_nnet_and_data_packets(blocking=True)

            ### Uncomment to print ops
            # ops = ops + 1
            # if time() - prevTime > 1.0:
            #     print('OPS: ', ops)
            #     ops = 0
            #     prevTime = time()

            packets_len = len(self.nnet_packets) + len(self.data_packets)
            if packets_len != 0:
                self.reset_process_wd()
            else:
                cur_time = monotonic()
                if cur_time > wd_cutoff:
                    print("process watchdog timeout")
                    os._exit(10)

            for _, nnet_packet in enumerate(self.nnet_packets):
                if args['verbose']: print_packet_info(nnet_packet, 'NNet')

                meta = nnet_packet.getMetadata()
                camera = 'rgb'
                if meta != None:
                    camera = meta.getCameraName()
                nnet_prev["nnet_source"][camera] = nnet_packet
                nnet_prev["entries_prev"][camera] = decode_nn(nnet_packet, config=config, NN_json=NN_json)
                frame_count['metaout'] += 1
                frame_count['nn'][camera] += 1

            for packet in self.data_packets:
                window_name = packet.stream_name
                if packet.stream_name not in stream_names:
                    continue  # skip streams that were automatically added
                if args['verbose']: print_packet_info(packet, packet.stream_name)
                packetData = packet.getData()
                if packetData is None:
                    print('Invalid packet data!')
                    continue
                elif packet.stream_name == 'previewout':
                    meta = packet.getMetadata()
                    camera = 'rgb'
                    if meta != None:
                        camera = meta.getCameraName()

                    window_name = 'previewout-' + camera
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
                    # change shape (3, 300, 300) -> (300, 300, 3)
                    data0 = packetData[0, :, :]
                    data1 = packetData[1, :, :]
                    data2 = packetData[2, :, :]
#                     print(packet.stream_name) # == previewout
                    
                    frame = cv2.merge([data0, data1, data2])
                    
                    
##################################################################################################
##################################################################################################
                    
                    
                    if nnet_prev["entries_prev"][camera] is not None:
                        
                        frame = show_nn(nnet_prev["entries_prev"][camera], frame, NN_json=NN_json, config=config)
                        temp = nnet_prev["entries_prev"][camera] # temp is a list of a dictionary
                        
#                         SEEx_frame = 
                        
                        # this will only handle one balloon I believe 
                        if len(temp) > 0:
                            dict_var = temp[0] # dictionary
                            object_id = dict_var["class_id"]
                            
                            if object_id == 0: # blue ballon for 0
                                balloon_color = "BLUE"
                                GPIO.output(17,True)
                                GPIO.output(4,False)
#                                 exit()
                                
                            if object_id == 1: # red balloon ffor 1
                                balloon_color = "RED"
                                GPIO.output(17,False)
                                GPIO.output(4,True)
#                                 exit()
                                
                                
                            x = dict_var["depth_x"]
                            y = dict_var["depth_y"]
                            z = dict_var["depth_z"]
                            
                            print(balloon_color,x,y,z)
                        else:
                            GPIO.output(17,False)
                            GPIO.output(4,False)
                            
                            
                            
##################################################################################################                          
##################################################################################################                       
                        
                        
                        if enable_object_tracker and tracklets is not None:
                            frame = show_tracklets(tracklets, frame, labels)
#                     cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
#                     cv2.putText(frame, "NN fps: " + str(frame_count_prev['nn'][camera]), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
#                     print("line 290")
#                     print(window_name)
                    
                    
                    # make copy of image
                    passed_image = np.copy(frame)
#                     
                    # Contours
#                     contourImage(passed_image)
#                     cv2.imshow("dev", image_with_lines) # show lines on a dev feed

                    # Lines
                    height = passed_image.shape[0]
                    width = passed_image.shape[1]
                    
                    # make a canny image
                    canny_image = canny(passed_image)
                    
                    # regions of interest
                    # must be an array of polygons
                    roi_left  = np.array([
                    [(0, height), (width, height), (int(width/2), height), (int(width/2),0)]
                    ])
                    
#                     roi_right = np.array([
#                     [(int(width/2), 0), (int(width/2), height), (width, height), (width,0)]
#                     ])
                    
                    
                    # left and right cropped images
                    cropped_left = region_of_interest(canny_image,roi_left)
#                     cropped_right = region_of_interest(canny_image, roi_right)
                    
#                     cv2.imshow("Cropped Left", cropped_left)
                    
                    # lines on left and right
                    left_lines = cv2.HoughLinesP(cropped_left, 100,np.pi/180, 100, np.array([]), minLineLength = 5, maxLineGap = 5)
#                     right_lines = cv2.HoughLinesP(cropped_right, 100,np.pi/180, 10, np.array([]), minLineLength = 10, maxLineGap = 5)

                    if np.size(left_lines) > 1:
                        # averaged lines
                        ave_left_line_image = average_slope_intercept(passed_image, left_lines)
    #                     ave_right_line_image = average_slope_intercept(passed_image, right_lines)
                        
                        # Line images
                        left_line_image = display_lines(passed_image, ave_left_line_image)
            
    #                     right_line_image = display_lines(passed_image,ave_right_line_image)

                        
                            # Display Lines
#                         print("left ave")
#                         left_combo_image = cv2.addWeighted(passed_image, 0.8, left_line_image,1,1)
#                         cv2.imshow("LEFT", left_combo_image)
                            
#                     if right_line_image is not None:
#                         right_combo_image = cv2.addWeighted(passed_image, 0.8, right_line_image,1,1)
#                         cv2.imshow("RIGHT", right_combo_image)
                        
                    cv2.imshow(window_name, passed_image) # show depthai OG feed
#######################################################################################################################                    
                    
                elif packet.stream_name in ['left', 'right', 'disparity', 'rectified_left', 'rectified_right']:
                    frame_bgr = packetData
                    if args['pointcloud'] and packet.stream_name == 'rectified_right':
                        right_rectified = packetData
                    cv2.putText(frame_bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    print("line 298")
                    
                    if args['draw_bb_depth']:
                        camera = args['cnn_camera']
                        if packet.stream_name == 'disparity':
                            if camera == 'left_right':
                                camera = 'right'
                        elif camera != 'rgb':
                            camera = packet.getMetadata().getCameraName()
                        if nnet_prev["entries_prev"][camera] is not None:
                            frame_bgr = show_nn(nnet_prev["entries_prev"][camera], frame_bgr, NN_json=NN_json, config=config, nn2depth=nn2depth)
                    cv2.imshow(window_name, frame_bgr)
                elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
                    frame = packetData

                    if len(frame.shape) == 2:
                        if frame.dtype == np.uint8:  # grayscale
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                            cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                            print("line 317")
                        else:  # uint16
                            if args['pointcloud'] and "depth" in stream_names and "rectified_right" in stream_names and right_rectified is not None:
                                try:
                                    from depthai_helpers.projector_3d import PointCloudVisualizer
                                except ImportError as e:
                                    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e} \033[0m ")
                                if pcl_converter is None:
                                    pcl_converter = PointCloudVisualizer(self.device.get_right_intrinsic(), 1280, 720)
                                right_rectified = cv2.flip(right_rectified, 1)
                                pcl_converter.rgbd_to_projection(frame, right_rectified)
                                pcl_converter.visualize_pcd()

                            frame = (65535 // frame).astype(np.uint8)
                            # colorize depth map, comment out code below to obtain grayscale
                            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                            # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                            cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                            print("line 336")
                    else:  # bgr
                        cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                        cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                        print("line 340")
                    if args['draw_bb_depth']:
                        camera = args['cnn_camera']
                        if camera == 'left_right':
                            camera = 'right'
                        if nnet_prev["entries_prev"][camera] is not None:
                            frame = show_nn(nnet_prev["entries_prev"][camera], frame, NN_json=NN_json, config=config, nn2depth=nn2depth)
                    cv2.imshow(window_name, frame)

                elif packet.stream_name == 'jpegout':
                    jpg = packetData
                    mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                    cv2.imshow('jpegout', mat)

                elif packet.stream_name == 'video':
                    videoFrame = packetData
                    videoFrame.tofile(video_file)
                    # mjpeg = packetData
                    # mat = cv2.imdecode(mjpeg, cv2.IMREAD_COLOR)
                    # cv2.imshow('mjpeg', mat)
                elif packet.stream_name == 'color':
                    meta = packet.getMetadata()
                    w = meta.getFrameWidth()
                    h = meta.getFrameHeight()
                    yuv420p = packetData.reshape((h * 3 // 2, w))
                    bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
                    scale = configMan.getColorPreviewScale()
                    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    cv2.putText(bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    cv2.putText(bgr, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    print("line 370")
                    cv2.imshow("color", bgr)

                elif packet.stream_name == 'meta_d2h':
                    str_ = packet.getDataAsStr()
                    dict_ = json.loads(str_)

                    print('meta_d2h Temp',
                          ' CSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['css']),
                          ' MSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['mss']),
                          ' UPA:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa0']),
                          ' DSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa1']))
                elif packet.stream_name == 'object_tracker':
                    tracklets = packet.getObjectTracker()

                frame_count[window_name] += 1

            t_curr = time()
            if t_start + 1.0 < t_curr:
                t_start = t_curr
                # print("metaout fps: " + str(frame_count_prev["metaout"]))

                stream_windows = []
                for s in stream_names:
                    if s == 'previewout':
                        for cam in NN_cams:
                            stream_windows.append(s + '-' + cam)
                            frame_count_prev['nn'][cam] = frame_count['nn'][cam]
                            frame_count['nn'][cam] = 0
                    else:
                        stream_windows.append(s)
                for w in stream_windows:
                    frame_count_prev[w] = frame_count[w]
                    frame_count[w] = 0

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            else:
                keypress_handler(self, key, stream_names)

        del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
        del self.device
        cv2.destroyAllWindows()

        # Close video output file if was opened
        if video_file is not None:
            video_file.close()

        print('py: DONE.')


if __name__ == "__main__":
    dai = DepthAI()
    dai.startLoop()