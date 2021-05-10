#!/usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


read_dir = os.path.join(sys.path[0], "../data/")
save_dir = os.path.join(sys.path[0], "../results/")

# Parse arguments for "-mat"
parser = argparse.ArgumentParser(description='Method of obtaining Transform matrix')
parser.add_argument("-mat", default="api")
args = parser.parse_args()
mat = args.mat

distorted = cv2.imread(read_dir + "distorted.jpg")

# Shape = (rows, cols, channels)
desired_rows, desired_cols = 600, 600

# Correspondence points: [x,y] Obtained manually from cv2.imshow
pts1 = np.float32([[0, 0],
                   [601, 61],
                   [61, 601]])
pts2 = np.float32([[0, 0],
                   [600, 0],
                   [0, 600]])

if mat == "api":
  H = cv2.getAffineTransform(pts1, pts2)

elif mat == "manual":
  ## TODO: Manually derive 2*3 transformation matrix H
  '''
  H*X = Y
  H: 2*3 Direct Linear Transform matrix. Since this is known to be shear transform, we manually append zero column at the end
  X: 2*2 Input image pixels in euclidean coordinates
  X_prime: 2*2 Output image pixels in euclidean coordinates
  Solution: H = X_prime.X^{-1} and append zero_column
  '''
  pts1 = np.float32([[601, 61],
                     [61, 601]])
  pts2 = np.float32([[600, 0],
                     [0, 600]])
  X = pts1.transpose()
  # I = np.ones((1,2))
  # X = np.row_stack((X, I))
  X_prime = pts2.transpose()
  H = X_prime@np.linalg.inv(X)
  zero_column = np.zeros((2,1))
  H = np.column_stack((H, zero_column))

else:
  print("Incorrect input for method. Aborting!")
  sys.exit(0)

original = cv2.warpAffine(distorted, H, (desired_cols, desired_rows)) 

cv2.imshow("Distorted Chessboard", distorted)
cv2.imshow("Original Chessboard", original)

cv2.waitKey(0)
cv2.destroyAllWindows()
