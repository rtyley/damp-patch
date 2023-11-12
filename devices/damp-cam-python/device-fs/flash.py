import sys
import os
import time

from DFRobot_GP8403 import *
from math import sin

DAC = DFRobot_GP8403(0x5f)  
while DAC.begin() != 0:
    print("init error")
    time.sleep(1)
print("init succeed")

#Set output range  
DAC.set_DAC_outrange(OUTPUT_RANGE_10V)

DAC.set_DAC_out_voltage(10000, CHANNEL1)
# time.sleep(100)
DAC.set_DAC_out_voltage(0, CHANNEL1)

for i in range(1000000):
  v = int(500+(499*sin(i/80)))
  # print(v)
  DAC.set_DAC_out_voltage(v, CHANNEL1)
  time.sleep(0.01)
