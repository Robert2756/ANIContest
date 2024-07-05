import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from MIFS import MIFS
from learner import MLP
from learner import ContestDataset

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init

def infer_target(input):
    return 1.0 if input[1] > 6 else 0.0


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier Initialization
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

# # MIFS algorithm
# # Read the data from the file
# path_input = "./data/ANI_Training.Input"
# path_target = "./data/ANI_Training.Label"
# df = pd.read_csv(path_input, header=None) # (3250, 250)
# target = pd.read_csv(path_target, header=None) # (3250, 1)

# print(np.array(df).shape)
# print(np.array(target).shape)

# # MIFS algorithm and mutual information
# '''
# Which input channels korrelate the most with the output channel? -> Mutual information
# Which input channels korrelate the most with the output channel and have the lowest redundancy? -> MIFS algorithm
# '''

# data = np.transpose(df) # (250,  3250)
# selected_features = MIFS(np.array(data), np.array(target), beta=0.5, max_feature_len=20)
# print("selected features: ", selected_features)
# time.sleep(600)

selected_features = [225, 182, 8, 166, 165, 164, 4, 220, 219, 86, 161, 160, 159, 158, 157, 156, 155, 154]
dataset = ContestDataset(selected_channels=selected_features)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = MLP(input_size=len(selected_features), hidden_size=512, output_size=1).to("cpu")
model.apply(initialize_weights)

# Loss and optimizer
criterion = nn.MSELoss()  # Example loss function (Mean Squared Error)
# criterion = nn.BCELoss()  # Binary Cross Entropy Loss

# Learning rate scheduler and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.8)

step = 0
epoch_step = 0
num_epochs = 250
loss_average = 0

print("Start training")
for epoch in tqdm(range(num_epochs)):
    for inputs, targets in dataloader:

        step += 1
        epoch_step += 1

        targets = targets.view(-1, 1)  # Reshape to (batch_size, 1)

        # Forward pass
        # print("Input: ", inputs)
        # print("Target: ", targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # print("Outputs: ", outputs)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()
        loss_average +=loss


    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss average: {loss_average/epoch_step}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    epoch_step = 0
    loss_average = 0


# Save the checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}

torch.save(checkpoint, 'checkpoint.pth')


# implement test dataset evaluation to get objective model performance measure
# save ckp

# loss average was needed because of the huge amount of steps per epoch and hence high loss difference between different batches (some very low losses, some very high -> only one per epoch was previously shown)-> smooth loss value across batch
# this is obviously helpful if the training data is very different from each other and is very spread/ inhomogenous


# Improvements