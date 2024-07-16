import time
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim

from learner import MLP

TH = 0.5

def test(model, selected_channels):

    PREDICTIONS = []

    # import test dataset
    path = "./data/ANI_Test.Input"
    df = pd.read_csv(path, header=None) # (1750, 250)

    data = np.array(df.T) # (250,  1750)
    data = data[selected_channels, ::]

    # Normalize each feature separately
    # 1. Compute min and max for each column
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)

    data = 2 * (data - min_vals) / (max_vals - min_vals) - 1 # (5, 1750)

    # iterate over test data
    for i in range(np.array(data).shape[1]):
        output = model(torch.tensor(data[:,i], dtype=torch.float32))

        # make outputs discrete
        if output>=0.5:
            PREDICTIONS.append(1)
        elif output<0.5:
            PREDICTIONS.append(0)
    
    # write predictions to output file
    with open('data/Viehweg_63304_0507.Label', 'w') as file:
        for number in PREDICTIONS:
            file.write(f"{number}\n")


selected_channels = [225, 156, 132, 125, 58, 151, 172, 120, 117]
model = MLP(input_size=len(selected_channels), hidden_size=512, output_size=1).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the checkpoint
checkpoint = torch.load('checkpoint_BER_0.25705_3000traindata.pth')

# Load the state_dict of the model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# inference
model.eval()

test(model=model, selected_channels=selected_channels)

