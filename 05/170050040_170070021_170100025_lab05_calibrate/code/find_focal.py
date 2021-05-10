import numpy as np
import cv2

path = "../data/calib_images/"

img1 = cv2.imread(path+"7.jpg")
img2 = cv2.imread(path+"8.jpg")
res = img1.shape[0]/600


gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

world_points = np.array([[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]],[[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0]]],dtype = np.float32)
#For image 6.jpg and 7.jpg
# img_points = np.array([[[567, 2821], [759, 2821], [943, 2813], [567, 2646], [751, 2629], [926, 2612]],[[793, 2345], [1010, 2379], [1202, 2395], [776, 2112], [985, 2145], [1177, 2187]]],dtype = np.float32)
#For image 7.jpg and 8.jpg
img_points = np.array([[[793, 2345], [1010, 2362], [1202, 2395], [776, 2103], [993, 2145], [1177, 2178]], [[509, 2345], [784, 2320], [1026, 2312], [500, 2162], [751, 2153], [985, 2145]]],dtype = np.float32)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, img2.shape[1::-1], None, None)
for p in img_points[0]:
    img1 = cv2.circle(img1, (int(p[0]),int(p[1])), radius=20, color=(0, 0, 255), thickness=-1)
for p in img_points[1]:
    img2 = cv2.circle(img2, (int(p[0]),int(p[1])), radius=20, color=(0, 0, 255), thickness=-1)

cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img1', (min(int(img1.shape[1]/res),600),600))
cv2.imshow("img1",img1)

cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img2', (min(int(img2.shape[1]/res),600),600))
cv2.imshow("img2",img2)
cv2.waitKey(0)

fx = (mtx[0][0]*5.5)/4608
fy = (mtx[1][1]*4.1)/2112

print("Focal lenghths of camera are Fx = ",fx,"Fy = ",fy)
