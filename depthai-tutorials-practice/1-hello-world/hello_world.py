import numpy as np # import numpy as np
import cv2 # import open CV
import depthai # import the depthai lib


# Define a pipeline
pipeline = depthai.Pipeline()


# Make a ColorCamera node
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300,300) # resize preview for mobilenet
cam_rgb.setInterleaved(False) #

# make a pipeline for the neural network 
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath("mobilenet-ssd.blob") # tell it what blob file to use


cam_rgb.preview.link(detection_nn.input) # link cam_rgb.preview output to detection_nn input


# XLink for communication between FROM device (camera) TO host (RPi)
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)


# Initialize the DepthAI Device
device = depthai.Device(pipeline)
device.startPipeline()