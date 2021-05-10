from utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pickle

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

data_loc = '../data/data_test.pkl'
filepath = '../results/bestModel.pt'
device = torch.device('cpu')
#device = torch.device('cuda:0')

def run_epoch(epoch_no, data_in,data_out, model,device, optimiser, batch_size=256,split='train'):
    train_set = myDataset(data_in,data_out)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True)
    total_loss,count = 0,0
    for x,y in train_loader:
        y_pred = model(x.to(device))
        loss = cal_mpjpe(y_pred,y.to(device))
        if (split=='train'):
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        count += x.shape[0]
        total_loss += float(loss.detach().cpu())*x.shape[0]
    return float(total_loss/count)
        
model = LiftModel().to(device)
model.load_state_dict(torch.load(filepath))
model.eval()
with open(data_loc,'rb') as f:
    data = pickle.load(f)
    
train_input = data['joint_2d_1']
# train_output = np.zeros((train_input.shape[0],train_input.shape[1],train_input.shape[2]))
train_output = data['joint_3d']

with torch.no_grad():
    loss = run_epoch(0, train_input,train_output,model, device,None, batch_size=64,split='val')
print (loss)
