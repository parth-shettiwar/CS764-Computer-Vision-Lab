import numpy as np
import argparse
import sys
file_name =  sys.argv[1]
a = np.loadtxt(file_name,delimiter=',')
m = int(input ("M = "))
n = int(input ("N = "))

a = np.repeat(a[:,:,np.newaxis,np.newaxis],m,axis=2)
a = np.repeat(a,n,axis=3)
print (a)
