import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from DFRobot_GP8403 import *

DAC = DFRobot_GP8403(0x5f)  
while DAC.begin() != 0:
    print("init error")
    time.sleep(1)
print("init succeed")

DAC.set_DAC_outrange(OUTPUT_RANGE_10V)
DAC.set_DAC_out_voltage(0, CHANNEL1)

# DAC.store()  # This currently leaves I2C in an unusable state - needs to go back to ALT0 
