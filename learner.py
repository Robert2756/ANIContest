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
        self.target = pd.read_csv(path_target, header=None) # (3250, 1)

        data = np.array(df.T) # (250,  3250)
        data = data[selected_channels, ::]

        # # Normalize the selected data to the range [-1, 1]
        # min_vals = data.min(axis=0, keepdims=True)
        # max_vals = data.max(axis=0, keepdims=True)
        # self.data = 2 * (data - min_vals) / (max_vals - min_vals) - 1 # [-1,1]

        # self.selected_channels = selected_channels

        # Normalize each feature separately
        # 1. Compute min and max for each column
        min_vals = np.min(data, axis=1, keepdims=True)
        max_vals = np.max(data, axis=1, keepdims=True)
        # print("min vals: ", min_vals)
        # print("min vals: ", max_vals)

        self.data = 2 * (data - min_vals) / (max_vals - min_vals) - 1
        # print("normalized data: ", normalized_data)

        time.sleep(10)
        
    def __len__(self):
        return self.data.shape[1] # number of examples in the dataset
    
    def __getitem__(self, idx):
        data = self.data[:, idx]
        target = self.target.iloc[idx, 0]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
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
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

        # Apply Xavier initialization to the weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        # print("x: ", x)
        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        # print("out: ", out)

        # y = self.sigmoid(out)
        # y = self.softmax(out)

        # y = F.softmax(out, dim=1)
        y = F.sigmoid(out)
        # print("y: ", y)
        return y