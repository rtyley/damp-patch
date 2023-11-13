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
import cv2
from credentials import adafruit_io_username, adafruit_io_key
from Adafruit_IO import Client

from DFRobot_GP8403 import *

DAC = DFRobot_GP8403(0x5f)
while DAC.begin() != 0:
  print("init error")
  time.sleep(1)
print("init succeed")

# Set output range
DAC.set_DAC_outrange(OUTPUT_RANGE_10V)

lens_pos = 0.8

exposure_time = 8500


def set_light(on: bool):
  print(f'on={on}')
  DAC.set_DAC_out_voltage(10000 if on else 0, CHANNEL1)


def upload_image(filepath: str):
  print(f'Uploading {filePath}')
  with open(filePath, 'rb') as f:
    r = requests.post(
      f'https://www.teleport.io/api/v1/frame-set/fetebzuxkyjl?apikey={teleport_api_key}',
      data=f, verify=False)
    print(f'Upload complete! {r.status_code} - {"GOOD" if r.status_code == 200 else "Malfunction?"}')


def find_bright_pixels(colour_image):
  gray_image = cv2.cvtColor(colour_image, cv2.COLOR_RGB2GRAY)
  # cv2.imshow('Grayscale', gray_image)
  threshold = 200
  ret, threshold_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_TOZERO)
  return threshold_image


def normalise(arr):
  min_val = numpy.min(arr)
  max_val = numpy.max(arr)
  print(f'min_val={min_val} max_val={max_val}')
  return (arr - min_val) / (max_val - min_val)


def report_score_of_image(colour_image):
  loaded_mask = normalise(cv2.cvtColor(cv2.imread("just_damp_never_bright.edited.png"), cv2.COLOR_BGR2GRAY))
  raw_score = (find_bright_pixels(colour_image) * loaded_mask).sum()
  print(raw_score)

  aio = Client(adafruit_io_username, adafruit_io_key)
  feed = aio.feeds('glisten')
  aio.send_data(feed.key, raw_score)


with (Picamera2() as picam):
  config = picam.create_still_configuration()
  picam.configure(config)
  picam.set_controls(
    {"AfMode": controls.AfModeEnum.Manual, "LensPosition": lens_pos, "ExposureTime": exposure_time,
     "AnalogueGain": 1.0})
  picam.start()

  now = datetime.datetime.now().replace(microsecond=0)
  ts = now.isoformat().replace(':', '-')
  print(ts)
  fileName = f"{ts}.exposure-{exposure_time}.jpg"
  tmpDir = f"/tmp/damp-cam-tmp/{now.strftime('%Y-%m-%d/%H')}"
  # with tempfile.TemporaryDirectory() as tmpDir:
  print(f'Folder directory: {tmpDir}')
  Path(tmpDir).mkdir(parents=True, exist_ok=True)
  filePath = os.path.join(tmpDir, fileName)

  set_light(on=True)
  try:
    time.sleep(2)
    array = picam.capture_array()
  finally:
    set_light(on=False)

  image = Image.fromarray(array).rotate(-90, expand=True).crop((0, 0, 2592, 4320))
  image.save(filePath)

  report_score_of_image(numpy.array(image))

  upload_image(filePath)

