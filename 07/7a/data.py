from __future__ import print_function, division
import torch.nn as nn
import numpy as np
import pickle 
import os
import torch

with open('data_train_lift.pkl', 'rb') as f:
    dat = pickle.load(f)


sha = dat["focal_len_1"].shape[0]
train = dat.copy()
test = dat.copy()
train["joint_2d_1"] = train["joint_2d_1"][:int(.80*sha)]
train["joint_3d"] = train["joint_3d"][:int(.80*sha)]
train["focal_len_1"] = train["focal_len_1"][:int(.80*sha)]

test["joint_2d_1"] = test["joint_2d_1"][int(.80*sha):]
test["joint_3d"] = test["joint_3d"][int(.80*sha):]
test["focal_len_1"] = test["focal_len_1"][int(.80*sha):]


with open('data_trainN_lift.pkl', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL) 

with open('data_test_lift.pkl', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)        