#!/usr/bin/env python3

import cv2
import sys

cv2.namedWindow("Slideshow")

filedir = sys.argv[1]
img = cv2.imread(filedir + "/display00.jpg") # load initial image
i = 0
n = 5

while True:
    cv2.imshow("Slideshow", img)

    key = chr(cv2.waitKey(0))
    if key == 'n': # Next
      i = (i+1) % n
    elif key == 'p': # Previous
      i = (i-1) % n
    else:  #use any other key to escape
      break
    img = cv2.imread(filedir + f"/display0{i}.jpg")

cv2.destroyAllWindows()