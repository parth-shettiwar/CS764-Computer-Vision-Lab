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

weak = True

if weak:
    data_loc = '../data/data_train.pkl'
else :
    data_loc = '../data/data_train_lift.pkl'
    
filepath = '../results/bestModel.pt'
#device = torch.device('cpu')
device = torch.device('cuda:0')

def run_epoch_weak(epoch_no, data_in1, data_in2, data_rot, data_trans, model,device, optimiser, batch_size=256,split='train'):
    train_set = myDatasetWeak(data_in1,data_in2,data_rot,data_trans)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True)
    total_loss,count = 0,0
    for x1,x2,rot,trans in train_loader:
        y_pred1 = model(x1.to(device))
        y_pred2 = model(x2.to(device))
        y_pred_other = transform(y_pred1,trans.to(device),rot.to(device))
        loss = cal_mpjpe(y_pred2,y_pred_other)

        if (split=='train'):
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        count += x1.shape[0]
        total_loss += float(loss.detach().cpu())*x1.shape[0]
    return float(total_loss/count)


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
lr = 1e-3
optim = torch.optim.Adam(model.parameters(),lr=lr)
max_epoch = 10
with open(data_loc,'rb') as f:
    data = pickle.load(f)

num_examples = data['joint_2d_1'].shape[0]

if weak:
    train_input1 = data['joint_2d_1'][:-int(num_examples/10)]
    train_input2 = data['joint_2d_2'][:-int(num_examples/10)]
    train_rot = data['rot'][:-int(num_examples/10)]
    train_trans = data['transl'][:-int(num_examples/10)]
else :
    train_input = data['joint_2d_1'][:-int(num_examples/10)]
    train_output = data['joint_3d'][:-int(num_examples/10)]

val_input = data['joint_2d_1'][-int(num_examples/10):]
val_output = data['joint_3d'][-int(num_examples/10):]

train_losses,val_losses = [],[]
for epoch_no in range(max_epoch):
    if (weak):
        train_loss = run_epoch_weak(epoch_no, train_input1,train_input2,train_rot,train_trans,model, device,optim, batch_size=64)
    else:
        train_loss = run_epoch(epoch_no, train_input,train_output,model, device,optim, batch_size=64)
        
    with torch.no_grad():
        model.eval()
        val_loss = run_epoch(epoch_no, val_input,val_output,model, device,optim, batch_size=64,split='val')
        model.train()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print (epoch_no,train_loss,val_loss)
    
torch.save(model.state_dict(), filepath)
