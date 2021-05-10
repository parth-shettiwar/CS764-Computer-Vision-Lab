import numpy as np
import cv2
import argparse,copy
np.random.seed(0)

def sort(arr):
    # hardcoded for now
    arr = arr[np.argsort(arr[:,1])]
    arr[:3] = arr[:3][np.argsort(arr[:3,0])]
    arr[3:] = arr[3:][np.argsort(arr[3:,0])]
    return arr


parser = argparse.ArgumentParser()
parser.add_argument("-f", action="store")
args = parser.parse_args()
image_shape_ = [(700, 900),(900, 700),(900, 900),(700, 700),(650, 750),(750, 650),(1000, 1000),(1200,800),(900,1100),(2000, 2000)]

for image_shape in image_shape_:
     arr_order = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],dtype=np.float32)
     objpoints = np.repeat(arr_order[None,:,:],2,axis=0).astype(np.float32)

     points = (np.loadtxt(args.f)).astype(np.float)
     pts1 = points[:6,:]
     pts2 = points[6:12,:]
     imgpoints = np.array([sort(pts1),sort(pts2)], dtype=np.float32)

     ret, mtx, dist, rvecs, tvecs = \
         cv2.calibrateCamera(objpoints, imgpoints,
                             image_shape, None, None,flags = 
                              (cv2.CALIB_ZERO_TANGENT_DIST+ cv2.CALIB_FIX_K1+ cv2.CALIB_FIX_K2+ cv2.CALIB_FIX_K3))


     mean_error = 0
     for i in range(len(imgpoints)):
         imgpoints2, _ = cv2.projectPoints(arr_order, rvecs[i], tvecs[i], mtx, dist)
         error = cv2.norm(imgpoints[i],imgpoints2[:,0,:], cv2.NORM_L2)/len(imgpoints2)
         mean_error += error

        
     print (image_shape,mean_error/len(objpoints))

