import cv2
import numpy as np
from credentials import adafruit_io_username, adafruit_io_key
from Adafruit_IO import Client

known_dry = "/tmp/damp-cam-tmp/2023-11-12/21/2023-11-12T21-38-20.exposure-8500.jpg"
known_start_of_flash = "/tmp/damp-cam-tmp/2023-11-12/19/2023-11-12T19-38-20.exposure-8500.jpg"
known_wet = "/tmp/damp-cam-tmp/2023-11-12/20/2023-11-12T20-00-20.exposure-8500.jpg"


def original_image():
  return cv2.cvtColor(cv2.imread(known_dry), cv2.COLOR_BGR2RGB)


def find_bright_pixels(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # cv2.imshow('Grayscale', gray_image)
  threshold = 200
  ret, threshold_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_TOZERO)
  return threshold_image


def normalise(arr):
  min_val = np.min(arr)
  max_val = np.max(arr)
  print(f'min_val={min_val} max_val={max_val}')
  return (arr - min_val) / (max_val - min_val)


def report_score_of_image(image):
  loaded_mask = normalise(cv2.cvtColor(cv2.imread("just_damp_never_bright.edited.png"), cv2.COLOR_BGR2GRAY))
  raw_score = (find_bright_pixels(image) * loaded_mask).sum()
  print(raw_score)

  aio = Client(adafruit_io_username, adafruit_io_key)
  feed = aio.feeds('beta-glisten')
  aio.send_data(feed.key, raw_score)


photo_image = original_image()

report_score_of_image(photo_image)
