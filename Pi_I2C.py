import smbus
import time
bus = smbus.SMBus(1)
DEVICE_ADDRESS = 0x08
def StringToBytes(val):
    value = [];
    for i in  val:
        value.append(ord(i))
        return value
#Turning it on
data = StringToBytes('ON')
print(data)
bus.write_i2c_block_data(DEVICE_ADDRESS,0x00,data)
# bus.write_i2c_block_data(DEVICE_ADDRESS,0x00,data2)
#bus.write_byte_data(DEVICE_ADDRESS,0,StringToBytes('O'))
#bus.write_byte_data(DEVICE_ADDRESS,0,StringToBytes('N'))
#Turning it off
# elif i = 0:
#     bus.write_byte_data(DEVICE_ADDRESS,0,'O')
#     bus.write_byte_data(DEVICE_ADDRESS,0,'F')
#     bus.write_byte_data(DEVICE_ADDRESS,0,'F')
word = bus.read_byte_data(DEVICE_ADDRESS,0)
print(word)