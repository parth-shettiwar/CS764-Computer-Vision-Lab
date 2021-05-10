#!/usr/bin/env python3

# x 512 , y 437 A 50 
# phi 3*pi

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys,copy

read_img = "../data/piece/brick.png"
save_img = "../results/piece-affine-results/brick.png"
image  = cv2.imread(read_img)
cv2.imshow("d",image)
cv2.waitKey(0)
print("fff")
rows, cols = image.shape[0], image.shape[1]
desired_rows, desired_cols = 337, 25
cycles = 3*np.pi
final = []

for colId in range(20):
    left_bottom = [0,0]
    right_bottom = [0,int(cols/20)]
    left_top = [rows,0]
    right_top = [rows,int(cols/20)]
    changeYleft = +50*(1+np.sin(colId*cycles/20))
    changeYright = +50*(1+np.sin((colId+1)*cycles/20))
    dest_points = np.array([left_bottom,left_top,right_bottom,right_top]).astype(np.float)
    src_points = copy.deepcopy(dest_points)
    src_points[:2,0] += changeYleft
    src_points[2:,0] += changeYright
    dest_points = np.float32(dest_points)
    src_points = np.float32(src_points)
    src_points = np.flip(src_points,axis=1)
    dest_points = np.flip(dest_points,axis=1)
    H = cv2.getAffineTransform(src_points[:3],dest_points[:3])
    original = cv2.warpAffine(image[:,int(colId*cols/20):int((colId+1)*cols/20),:], H, (desired_cols, desired_rows))
    final.append(original)
cv2.imwrite("../results/piece-affine-results/brick.png", np.concatenate(final,axis=1))
