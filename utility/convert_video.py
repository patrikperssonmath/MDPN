

import numpy as np
import cv2
import os
import shutil

path = "/data/colmap/skrylle"

cap = cv2.VideoCapture(os.path.join(path, "20200822_163141.mp4"))

path_image = os.path.join(path, "images")

if os.path.exists(path_image):
    shutil.rmtree(path_image)

os.makedirs(path_image)

count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(os.path.join(path_image, "{0:06d}.png").format(count), frame)

        count += 1
    else:
        break


# When everything done, release the capture
cap.release()
