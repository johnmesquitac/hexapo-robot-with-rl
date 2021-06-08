import serial
from digi.xbee.devices import XBeeDevice
from digi.xbee.devices import *
from digi.xbee.util import *
from digi.xbee.packets import *

NEW_TIMEOUT_FOR_SYNC_OPERATIONS = 5 # 5 seconds 

device = XBeeDevice("COM4", 9600)
device.open()
remote_device = RemoteXBeeDevice(device,XBee64BitAddress.from_hex_string("0013A20040631612"))
# Send data using the remote object.
robot_info = device.read_data()            
while robot_info is None:
        robot_info = device.read_data()
robot_info = robot_info.data.decode()
if 'Posicao' in robot_info:
    device.send_data(remote_device, "0")
    xbee_message = device.read_data()
    while xbee_message is None:
        xbee_message = device.read_data()
    print(xbee_message.data.decode())
