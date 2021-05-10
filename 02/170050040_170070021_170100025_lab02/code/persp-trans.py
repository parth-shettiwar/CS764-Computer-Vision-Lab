import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import cv2

image = cv2.imread('../data/obelisk.png')

corners = np.float32([[218,239],[483,189],[988,829],[703,1024]])
new_corners = np.float32([[0, 385],[0,0], [512, 0], [512, 385]])

mask = np.zeros(image.shape, dtype=np.uint8)
cv2.fillPoly(mask, pts=[corners.astype(np.int)], color=(255,255,255))

image = cv2.bitwise_and(image, mask)

num_frames = 100
wait_time = 100
list_corners = [(corners*(num_frames-x)+x*new_corners)/num_frames for x in range(0,num_frames+1)]

for this_corner in list_corners : 
    H_matrix = cv2.getPerspectiveTransform(corners, this_corner) 
    final = cv2.warpPerspective(image, H_matrix,  (1080,1080))
    cv2.imshow("obelisk",final)
    key = cv2.waitKey(wait_time)
    if key == ord('q'):
        break
