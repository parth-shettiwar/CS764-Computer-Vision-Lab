#!/usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

read_dir = os.path.join(sys.path[0], "../data/calib_images/")
save_dir = os.path.join(sys.path[0], "../results/")

img1 = cv2.imread(read_dir+"1.jpg")
img2 = cv2.imread(read_dir+"2.jpg")

## CAMERA CALIBRATION ROUTINE

objectPoints = np.array([[[3,0,0],[6,0,0],[9,0,0],[3,3,0],[6,3,0],[9,3,0]],[[3,0,0],[6,0,0],[9,0,0],[3,3,0],[6,3,0],[9,3,0]]],dtype = np.float32)

imagePoints = np.array([[[960, 2955], [1185, 2921], [1435, 2880], [909, 2679], [1135, 2654], [1377, 2629]],[[576, 2671], [776, 2704], [1018, 2754], [642, 2479], [851, 2512], [1068, 2562]]],dtype = np.float32)

ret, calib_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
  objectPoints, imagePoints, (img1.shape[:2])[::-1], None, None)
print("Intrinsic camera calibration matrix:")
print(calib_matrix)



## REPROJECTION ERROR ROUTINE
## Img 1
reproj_points, _ = cv2.projectPoints(objectPoints[0], rvecs[0], tvecs[0], calib_matrix, dist)
reproj_points = reproj_points[:,0,:]

print(f"Mean L2 re-projection error in image 1 = {np.linalg.norm(reproj_points-imagePoints[0])/6.:.3f}")

image_tuples = [tuple( map(int,i) ) for i in imagePoints[0]]
for pt in image_tuples:
  cv2.circle(img1, pt, 15, (255,0,255), 8)

cv2.imwrite(save_dir+"true-1.jpg", img1)

reproj_tuples = [tuple( map(int,i) ) for i in reproj_points]
for pt in reproj_tuples:
  cv2.drawMarker(img1, pt, (255,255,0),markerType=cv2.MARKER_STAR, 
    markerSize=40, thickness=2, line_type=cv2.LINE_AA)

cv2.imwrite(save_dir+"projected-1.jpg", img1)

cv2.namedWindow("Image 1", cv2.WINDOW_NORMAL)
cv2.imshow("Image 1", img1)


## Img 2
reproj_points, _ = cv2.projectPoints(objectPoints[1], rvecs[1], tvecs[1], calib_matrix, dist)
reproj_points = reproj_points[:,0,:]

print(f"Mean L2 re-projection error in image 2 = {np.linalg.norm(reproj_points-imagePoints[1])/6.:.3f}")

image_tuples = [tuple( map(int,i) ) for i in imagePoints[1]]
for pt in image_tuples:
  cv2.circle(img2, pt, 15, (255,0,255), 8)

cv2.imwrite(save_dir+"true-2.jpg", img2)

reproj_tuples = [tuple( map(int,i) ) for i in reproj_points]
for pt in reproj_tuples:
  cv2.drawMarker(img2, pt, (255,255,0),markerType=cv2.MARKER_STAR, 
    markerSize=40, thickness=2, line_type=cv2.LINE_AA)

cv2.imwrite(save_dir+"projected-2.jpg", img2)

cv2.namedWindow("Image 2", cv2.WINDOW_NORMAL)
cv2.imshow("Image 2", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()


