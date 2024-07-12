import time
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
from MIFS import mutual_infxy
from torch.utils.data import Dataset

import torch.nn.functional as F

path_input = "./data/ANI_Training.Input"
path_target = "./data/ANI_Training.Label"

class ContestDataset(Dataset):
    def __init__(self, selected_channels):
        df = pd.read_csv(path_input, header=None) # (3250, 250)
        df_train = df.iloc[0:2500, :]
        # print("df_train: ", np.array(df_train).shape)

        target_train = pd.read_csv(path_target, header=None) # (3250, 1)
        target_train = target_train.iloc[0:2500, :]
        self.target = target_train

        data = np.array(df_train.T) # (250,  3250)
        data = data[selected_channels, ::]

        # Normalize each feature separately
        min_vals = np.min(data, axis=1, keepdims=True)
        max_vals = np.max(data, axis=1, keepdims=True)
        # print("min vals: ", min_vals)
        # print("min vals: ", max_vals)

        self.data = 2 * (data - min_vals) / (max_vals - min_vals) - 1

        # for i, channel in enumerate(self.data):
        #     print(i)
        #     MI = mutual_infxy(np.array(channel), np.array(self.target))
        #     print(MI)
        # time.sleep(100)
        
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
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.tanh1 = nn.Tanh()
        self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(int(hidden_size/2))
        self.relu2 = nn.ReLU()
        # self.tanh2 = nn.Tanh()
        # self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, int(hidden_size/4))
        # self.tanh3 = nn.Tanh()
        self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(int(hidden_size/4), output_size)
        # self.relu4 = nn.ReLU()
        # self.fc5 = nn.Linear(hidden_size, hidden_size)
        # self.relu5 = nn.ReLU()
        # self.fc6 = nn.Linear(hidden_size, int(hidden_size/4))
        # self.relu6 = nn.ReLU()
        # self.fc7 = nn.Linear(int(hidden_size/4), output_size)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()
    
    def forward(self, x):
        # print("x: ", x)
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.tanh1(out)
        # out = self.dropout1(out)
        out = self.fc2(out)
        # out = self.tanh2(out)
        out = self.relu2(out)
        # out = self.dropout2(out)
        out = self.fc3(out)
        # out = self.tanh3(out)
        out = self.relu3(out)
        # out = self.dropout3(out)
        out = self.fc4(out)
        # out = self.relu4(out)
        # out = self.fc5(out)
        # out = self.relu5(out)
        # out = self.fc6(out)
        # out = self.relu6(out)
        # out = self.fc7(out)
        y = F.sigmoid(out)
        # print("y: ", y)
        return y