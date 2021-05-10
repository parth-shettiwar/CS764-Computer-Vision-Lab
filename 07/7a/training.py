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

def run_epoch(epoch_no, data, model, optimizer, scheduler, batch_size=64, split="train"):
    epoch_loss = 0
    for i,inp in enumerate(tqdm(data)):
        scheduler.step()
        coords2d, coords3d = inp
        coords2d = coords2d.to(device)
        coords3d = coords3d.to(device)
        coords2d = torch.flatten(coords2d,start_dim=1)
        coords3d = torch.flatten(coords3d,start_dim=1)
        optimizer.zero_grad()
        out = model(coords2d)
        coords3d = coords3d.reshape(batch_size,15,3)
        out = out.reshape(batch_size,15,3)
        loss = cal_mpjpe(out,coords3d)
        epoch_loss += loss 
        loss.backward()
        optimizer.step()
    return epoch_loss

class SkeletonData(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data["joint_2d_1"])


    def __getitem__(self, idx):
         coords2d = self.data["joint_2d_1"][idx]
         coords3d = self.data["joint_3d"][idx]
         return coords2d, coords3d
                                

class LiftModel(nn.Module):
    def __init__(self,n_blocks=2, hidden_layer=1024, dropout=0.1, output_nodes=15*3):
        super(LiftModel, self).__init__()
        self.n_blocks = n_blocks
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.output_nodes = output_nodes
        self.hidden1 = nn.Linear(15*2, self.hidden_layer)
        block = nn.Sequential(
            nn.Linear(self.hidden_layer,self.hidden_layer),nn.BatchNorm1d(self.hidden_layer),nn.ReLU(),nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer,self.hidden_layer),nn.BatchNorm1d(self.hidden_layer),nn.ReLU(),nn.Dropout(self.dropout)
        )
        for i in range(self.n_blocks):
            self.__setattr__('res-block_%d'%(i), block)
        self.output = nn.Linear(self.hidden_layer,output_nodes)

    def forward(self,inp):
        hid_out = self.hidden1(inp)
        x = hid_out
        x = self.__getattr__('res-block_%d'%(0))(x)
        x = x + hid_out
        intermediate = x    
        new = self.__getattr__('res-block_%d'%(1))(intermediate)
        final = x + new
        out = self.output(final)    
        return out


def main():
    model = LiftModel(dropout = 0.5).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.1, betas=(0.5, 0.999))
    scheduler = StepLR(optimizer, step_size=200, gamma=0.3)

    with open('data_trainN_lift.pkl', 'rb') as f:
        dat = pickle.load(f)
    Batch_size = 32
    dataset = SkeletonData(dat)  
    train_loader = torch.utils.data.DataLoader(
                  dataset, batch_size = Batch_size,
                  shuffle=True,
                  drop_last=True)

    best_loss = 100000
    num_epochs = 5

    for i in tqdm(range(num_epochs)):
        loss = run_epoch(i, train_loader, model, optimizer, scheduler, Batch_size, split="train")
        print("loss for epoch",i," = ",loss/len(train_loader))
        if(loss<best_loss):
            best_loss = loss
            torch.save(model.state_dict(),"model_best.pt")

if __name__ == "__main__":
    main()