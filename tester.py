import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim

from learner import MLP

def test(model, selected_channels):

    # import test dataset
    path = "./data/ANI_Test.Input"
    df = pd.read_csv(path, header=None) # (1750, 250)

    print("df: ", df.shape)

    data = np.array(df.T) # (250,  1750)
    data = data[selected_channels, ::]

    # Normalize each feature separately
    # 1. Compute min and max for each column
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    # print("min vals: ", min_vals)
    # print("min vals: ", max_vals)

    data = 2 * (data - min_vals) / (max_vals - min_vals) - 1 # 
    print("Data shape: ", np.array(data).shape)



selected_channels = [225, 182, 8, 166, 165]
model = MLP(input_size=len(selected_channels), hidden_size=64, output_size=1).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the checkpoint
checkpoint = torch.load('checkpoint.pth')

# Load the state_dict of the model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# inference
model.eval()

test(model=model, selected_channels=selected_channels)

