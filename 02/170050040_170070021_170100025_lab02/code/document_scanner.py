import numpy as np 
import cv2 
import argparse

parser = argparse.ArgumentParser(description='doc scan')
parser.add_argument('-i')
args = parser.parse_args()
image = cv2.imread(args.i)

# a=image.shape[0]
# b=image.shape[1]
# aspect = b/a

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary_image = cv2.threshold(grey_image, 120, 255, cv2.THRESH_BINARY)[1]
contors,_ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
biggest_cont = max(contors, key=cv2.contourArea)

br = -1e10
tl = 1e10
tr = -1e10
bl = -1e10

for i in range(len(biggest_cont)):
    if((biggest_cont[i,0,0]+biggest_cont[i,0,1])>br):
        br_val = biggest_cont[i,0]
        br = biggest_cont[i,0,0]+biggest_cont[i,0,1]
    if((biggest_cont[i,0,0]+biggest_cont[i,0,1])<tl):
        tl_val = biggest_cont[i,0]
        tl = biggest_cont[i,0,0]+biggest_cont[i,0,1]
    if((biggest_cont[i,0,0]-biggest_cont[i,0,1])>tr):
        tr_val = biggest_cont[i,0]
        tr = biggest_cont[i,0,0]-biggest_cont[i,0,1]
    if((-biggest_cont[i,0,0]+biggest_cont[i,0,1])>bl):
        bl_val = biggest_cont[i,0]  
        bl = -biggest_cont[i,0,0]+biggest_cont[i,0,1]   
                         
corners = np.float32([tl_val, tr_val, br_val, bl_val]) 
new_corners = np.float32([[0, 0], [400, 0], [400, 600], [0, 600]]) 

H_matrix = cv2.getPerspectiveTransform(corners, new_corners) 

final = cv2.warpPerspective(image, H_matrix, (400, 600)) 

cv2.imshow("document_scan",final)
cv2.waitKey(0)
