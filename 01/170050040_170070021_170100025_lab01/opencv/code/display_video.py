#!/usr/bin/env python3

import numpy as np
import cv2
import sys

print(len(sys.argv))
if len(sys.argv) < 2:
  cap = cv2.VideoCapture(0)
else:
  cap = cv2.VideoCapture(sys.argv[1])

while(True):
  # Capture frame-by-frame
  _, frame = cap.read()
  width, height, _ = np.shape(frame)

  # Display the resulting frame
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.rectangle(frame, (height-170,50), (height-15,15), (255,255,255), -1)
  cv2.putText(frame, 'Group 7', (height-150,40), font, 1, (0,0,0), 2, cv2.LINE_AA)
  # Display the color frame
  cv2.imshow('Color',frame)
  cv2.moveWindow('Color',50,50)

  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Display the gray frame
  cv2.imshow('Grayscale', gray_frame)
  cv2.moveWindow('Grayscale',800,50)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
