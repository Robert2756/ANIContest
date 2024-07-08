import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from MIFS import MIFS
from MIFS import mutual_infxy
from learner import MLP
from learner import ContestDataset
from tester import test

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init

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
# selected_features = MIFS(np.array(data), np.array(target), beta=0.1, max_feature_len=20)
# print("selected features: ", selected_features)
# time.sleep(600)

# selected_features = [225, 8, 181, 180, 179, 178, 177, 176, 175]
# selected_features = [i for i in range(0,250)]
selected_features = [225, 156, 8, 80] # , 170]

dataset = ContestDataset(selected_channels=selected_features)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = MLP(input_size=len(selected_features), hidden_size=64, output_size=1).to("cpu")

# Loss and optimizer
criterion = nn.MSELoss()  # Example loss function (Mean Squared Error)
# criterion = nn.BCELoss()  # Binary Cross Entropy Loss

# Learning rate scheduler and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.8)

step = 0
epoch_step = 0
num_epochs = 1
loss_average = 0
BERS = []
losses = []

print("Start training")

for epoch in tqdm(range(num_epochs)):
    for inputs, targets in dataloader:

        step += 1
        epoch_step += 1

        targets = targets.view(-1, 1)  # Reshape to (batch_size, 1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()
        loss_average +=loss

    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss average: {loss_average/epoch_step}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        BER = test(model=model, selected_channels=selected_features)
        BERS.append(BER)
        losses.append(loss_average.detach().numpy()/epoch_step)
    epoch_step = 0
    loss_average = 0


# Save the checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}

torch.save(checkpoint, 'checkpoint.pth')

print(losses)
fig, axes = plt.subplots(ncols=1, nrows=2)
axes[0].plot(np.linspace(1, num_epochs, num_epochs), BERS)
axes[0].set_title('BER per epoch')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('BER')
axes[1].plot(np.linspace(1, num_epochs, num_epochs), losses)
axes[1].set_title('Loss per epoch')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
plt.show()
