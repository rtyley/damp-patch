import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

images_folder = '/Users/roberto/development/time-lapse-dump'

image_files = [join(images_folder, f) for f in listdir(images_folder) if isfile(join(images_folder, f))]

print(f'Number of images {len(image_files)}')
print(image_files[0])

# Load the input image 
# wet_image = '2023-10-19T19_35_33+01_00'
# wetish_image = '2023-10-20T00_40_23+01_00'
# not_wet_image = '2023-10-21T21_49_23+01_00'

for image_file in image_files:
    image = cv2.imread(image_file)
    # cv2.imshow('Original', image)
    # cv2.waitKey(0)

    # Use the cvtColor() function to grayscale the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grayscale', gray_image)

    threshold = 224

    ret, threshold_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_TOZERO)
    titles = ['Original Image', 'TOZERO']
    images = [gray_image, threshold_image]

    # count number of white pixels
    mean = np.mean(threshold_image)
    print(f"{image_file.split('/')[5].split('.')[0]},{round(100*mean)}")

# for i in range(len(images)):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
#
# plt.show()
#
# cv2.waitKey(0)
#
# # Window shown waits for any key pressing event
# cv2.destroyAllWindows()
