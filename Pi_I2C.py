#import libraries
import smbus
import time


# I2C device 
bus = smbus.SMBus(1)
DEVICE_ADDRESS = 0x08

# Functions
def StringToBytes(val):
    value = [];
    for i in  val:
        value.append(ord(i))
#         print(value.append(ord(i)))
    
    return value
    
    
    
#Turning it on
data = StringToBytes('OFF')
print(data)
# print(type(data))
bus.write_i2c_block_data(DEVICE_ADDRESS,0x00,data)