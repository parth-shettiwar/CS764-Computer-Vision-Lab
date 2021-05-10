import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pickle

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(p=dropout))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(p=dropout))
        
    def forward(self,x):
        return x+self.layer2(self.layer1(x))

class LiftModel(nn.Module):
    def __init__(self,n_blocks=2, hidden_layer=1024, dropout=0.5, output_nodes=15*3):
        super(LiftModel, self).__init__()
        input_nodes = int(output_nodes*2/3)
        self.input_map = nn.Linear(input_nodes,hidden_layer)
        self.output_map = nn.Linear(hidden_layer,output_nodes)
        self.blocks = []
        for _ in range(n_blocks):
            self.blocks.append(ResidualBlock(hidden_layer,dropout))
        self.blocks = nn.ModuleList(self.blocks)
        

    def forward(self,poses):
        poses = self.input_map(poses.view(poses.shape[0],-1))
        for block in self.blocks:
            poses = block(poses)
        poses = self.output_map(poses).view(poses.shape[0],-1,3)
        return poses
    
def cal_mpjpe(pose_1, pose_2, avg=True):
    n_joints = pose_1.shape[1]
    batch_size = pose_1.shape[0]
    diff = pose_1-pose_2
    diff_sq = diff ** 2
    dist_per_joint = torch.sqrt(torch.sum(diff_sq, axis=2))
    dist_per_sample = torch.mean(dist_per_joint, axis=1)
    if avg is True:
        dist_avg = torch.mean(dist_per_sample)
    else:
        dist_avg = dist_per_sample
    return dist_avg

class myDataset(torch.utils.data.Dataset):
    def __init__(self,data_in,data_out):
        self.inputs = data_in
        self.outputs = data_out
    def __getitem__(self,index):
        return torch.FloatTensor(self.inputs[index]),torch.FloatTensor(self.outputs[index])
    def __len__(self):
        return self.inputs.shape[0]

class myDatasetWeak(torch.utils.data.Dataset):
    def __init__(self,data_in1,data_in2,data_rot,data_trans):
        self.inputs1 = data_in1
        self.inputs2 = data_in2
        self.rot = data_rot
        self.trans = data_trans
    def __getitem__(self,index):
        return torch.FloatTensor(self.inputs1[index]),torch.FloatTensor(self.inputs2[index]),torch.FloatTensor(self.rot[index]),torch.FloatTensor(self.trans[index])
    def __len__(self):
        return self.inputs1.shape[0]
    
    
def transform(view1,trans,rot):
    view1 = torch.matmul(rot.view(rot.shape[0],1,3,3),view1[:,:,:,None])[:,:,:,0]
    view1 = view1+trans[:,None,:]
    return view1

