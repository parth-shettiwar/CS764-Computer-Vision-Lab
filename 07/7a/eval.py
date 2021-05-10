from __future__ import print_function, division
import torch.nn as nn
import numpy as np
import pickle 
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from training import LiftModel, cal_mpjpe, SkeletonData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cal_mpjpe(pose_1, pose_2, avg=True):
  
    diff = pose_1-pose_2
    diff_sq = diff ** 2
    dist_per_joint = torch.sqrt(torch.sum(diff_sq, axis=2))    
    dist_per_sample = torch.mean(dist_per_joint, axis=1)
    if avg is True:
        dist_avg = torch.mean(dist_per_sample)
    else:
        dist_avg = dist_per_sample
    return dist_avg

def run_epoch_test( data, model, batch_size=64, split="test"):
    epoch_loss = 0
    with torch.no_grad():
      for i,inp in enumerate(tqdm(data)):
          coords2d, coords3d = inp
          coords2d = coords2d.to(device)
          coords3d = coords3d.to(device)
          coords2d = torch.flatten(coords2d,start_dim=1)
          coords3d = torch.flatten(coords3d,start_dim=1)
          out = model(coords2d)
          coords3d = coords3d.reshape(batch_size,15,3)
          out = out.reshape(batch_size,15,3)
          loss = cal_mpjpe(out,coords3d)
          epoch_loss += loss 
       
        
    return epoch_loss
                                

model = LiftModel(dropout = 0).to(device)
model.load_state_dict(torch.load("model_best.pt"))
model.eval()

with open('data_test_lift.pkl', 'rb') as f:
    dat = pickle.load(f)
Batch_size = 32
dataset = SkeletonData(dat)  
test_loader = torch.utils.data.DataLoader(
              dataset, batch_size = Batch_size,
              shuffle=False,
              drop_last=True)


loss = run_epoch_test( test_loader, model, Batch_size, split="test")
print("average test loss",loss/len(test_loader))
 
