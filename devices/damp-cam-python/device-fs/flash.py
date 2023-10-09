import sys
import os
import time

from DFRobot_GP8403 import *

DAC = DFRobot_GP8403(0x5f)  
while DAC.begin() != 0:
    print("init error")
    time.sleep(1)
print("init succeed")

#Set output range  
DAC.set_DAC_outrange(OUTPUT_RANGE_10V)

DAC.set_DAC_out_voltage(10000, CHANNEL1)
time.sleep(1)
DAC.set_DAC_out_voltage(0, CHANNEL1)
