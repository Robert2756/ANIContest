import time
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
from torch.utils.data import Dataset

import torch.nn.functional as F

path_input = "./data/ANI_Training.Input"
path_target = "./data/ANI_Training.Label"

class ContestDataset(Dataset):
    def __init__(self, selected_channels):
        df = pd.read_csv(path_input, header=None) # (3250, 250)
        # df_train = df.iloc[0:2800, :]
        df_train = df
        print("df_train: ", np.array(df_train).shape)

        target_train = pd.read_csv(path_target, header=None) # (3250, 1)
        # target_train = target_train.iloc[0:2800, :]
        self.target = target_train

        data = np.array(df_train.T) # (250,  3250)
        data = data[selected_channels, ::]

        # Normalize each feature separately
        min_vals = np.min(data, axis=1, keepdims=True)
        max_vals = np.max(data, axis=1, keepdims=True)
        # print("min vals: ", min_vals)
        # print("min vals: ", max_vals)

        self.data = 2 * (data - min_vals) / (max_vals - min_vals) - 1
        
    def __len__(self):
        return self.data.shape[1] # number of examples in the dataset
    
    def __getitem__(self, idx):
        data = self.data[:, idx]
        target = self.target.iloc[idx, 0]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# class SimpleAttention(nn.Module):
#     def __init__(self, d=1):
#         super(SimpleAttention, self).__init__()
#         self.d = d
#         self.softmax = nn.Softmax(dim=-1)
        
#     def forward(self, inputs):
#         # Compute attention scores
#         attention_scores = torch.matmul(inputs, inputs.transpose(0, 1))
        
#         # Normalize with softmax along the last dimension
#         attention_weights = self.softmax(attention_scores)
        
#         # Weighted sum of input vectors
#         output = torch.matmul(attention_weights, inputs)
        
#         return output, attention_weights
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()
    
    def forward(self, x):
        # print("x: ", x)
        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # out = self.relu3(out)
        # out = self.fc4(out)
        y = F.sigmoid(out)
        # print("y: ", y)
        return y