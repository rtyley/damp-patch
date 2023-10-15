import RPi.GPIO as GPIO
import time
import datetime
from picamera2 import Picamera2, Preview
from libcamera import controls
import requests
import tempfile
import os
from PIL import Image
import numpy
from pathlib import Path
from credentials import teleport_api_key

from DFRobot_GP8403 import *

DAC = DFRobot_GP8403(0x5f)
while DAC.begin() != 0:
    print("init error")
    time.sleep(1)
print("init succeed")

# Set output range
DAC.set_DAC_outrange(OUTPUT_RANGE_10V)

lens_pos = 0.8

now = datetime.datetime.now().replace(microsecond=0)
ts = now.isoformat().replace(':', '-')
print(ts)

exposure_time = 8500

fileName = f"{ts}.exposure-{exposure_time}.jpg"
tmpDir = f"/tmp/damp-cam-tmp/{now.strftime('%Y-%m-%d/%H')}"
# with tempfile.TemporaryDirectory() as tmpDir:
print(f'Folder directory: {tmpDir}')
Path(tmpDir).mkdir(parents=True, exist_ok=True)


def set_light(on: bool):
    print(f'on={on}')
    DAC.set_DAC_out_voltage(10000 if on else 0, CHANNEL1)


def upload_image(filepath: str):
    print(f'Uploading {filePath}')
    with open(filePath, 'rb') as f:
        r = requests.post(
            f'https://www.teleport.io/api/v1/frame-set/fetebzuxkyjl?apikey=${teleport_api_key}',
            data=f)
        print(f'Upload complete! {r.status_code}')


with Picamera2() as picam:
    config = picam.create_still_configuration()
    picam.configure(config)
    picam.set_controls(
        {"AfMode": controls.AfModeEnum.Manual, "LensPosition": lens_pos, "ExposureTime": exposure_time,
         "AnalogueGain": 1.0})
    picam.start()

    filePath = os.path.join(tmpDir, fileName)

    set_light(on=True)
    try:
        time.sleep(2)
        array = picam.capture_array()
    finally:
        set_light(on=False)

    image = Image.fromarray(array)
    image.rotate(-90, expand=True).crop((0, 0, 2592, 4320)).save(filePath)

    upload_image(filePath)
