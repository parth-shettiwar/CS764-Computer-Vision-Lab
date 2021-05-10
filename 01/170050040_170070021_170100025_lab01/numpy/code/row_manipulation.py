import numpy as np
import argparse

def crop_array(arr_2d, offset_height, offset_width, target_height, target_width):
    return arr_2d[offset_width:offset_width+target_width,offset_height:offset_height+target_height]

def pad_array(arr):
    arr_out = np.zeros((arr.shape[0]+4,arr.shape[1]+4))+0.5
    arr_out[2:arr_out.shape[0]-2,2:arr_out.shape[1]-2] = arr
    return arr_out

parser = argparse.ArgumentParser()
parser.add_argument('--N')

args = parser.parse_args()

N = int(args.N)

a = np.eye(N)
a = np.concatenate([a[::2,:],a[1::2,:]],axis=0)
cropped = crop_array(a,1,1,2,2)
padded = pad_array(cropped)
concatenated = np.concatenate([padded,padded],axis=1)
print ("Original array : ")
print (a)
print ("Cropped array : ")
print (cropped)
print ("Padded array : ")
print (padded)
print ("Concatenated array : shape = ",concatenated.shape)
print (concatenated)
