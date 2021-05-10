import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

file_name =  sys.argv[1]
a = np.loadtxt(file_name,delimiter=',')

outfile = '../results/out.png'

a_mean = np.mean(a,axis=0,keepdims=True)
a_std = np.std(a,axis=0,keepdims=True)
a = (a-a_mean)/a_std
cov = np.cov(a.T)
evalue , eigvec = np.linalg.eig(cov)
values = np.argsort(evalue)[-2:]
proj = eigvec[:,values]
a_pca = a.dot(proj)
# Verification code
# recovered  = a_pca@(proj.T)
#print (a_pca.shape)
#print (np.allclose(recovered,a))
plt.scatter(a_pca[:,0],a_pca[:,1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.savefig(outfile)
