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

# selected_features = [225, 8, 181, 180, 179, 178, 177, 176, 175]
# selected_features = [i for i in range(0,250)]

BERS_AV = []
BER = 1
REPETITION = 3

for  i in range(0, 250):
# while BER>0.28:or i==156 or i==8 or i==80
    print("i: ", i)

    if i==225 or i==156 or i==132 or i==125:
        BERS_AV.append(1)
        continue

    selected_features = [225, 156, 132, 125, i]
    num_epochs = 1
    BERS_rep = []

    # # train five times and average the BER value
    for rep in range(0,REPETITION):

        dataset = ContestDataset(selected_channels=selected_features)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = MLP(input_size=len(selected_features), hidden_size=32, output_size=1).to("cpu")

        # Loss and optimizer
        criterion = nn.MSELoss()  # Example loss function (Mean Squared Error)
        # criterion = nn.BCELoss()  # Binary Cross Entropy Loss

        # Learning rate scheduler and optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        step = 0
        epoch_step = 0
        losses = []
        loss_average = 0
        print("rep: ", rep)

        for epoch in tqdm(range(num_epochs)):
            for inputs, targets in dataloader:

                step += 1
                epoch_step += 1

                targets = targets.view(-1, 1)  # Reshape to (batch_size, 1)

                # Forward pass
                # print(inputs)
                outputs = model(inputs)
                # print("outputs: ", outputs)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Step the scheduler
                loss_average +=loss

            if (epoch+1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss average: {loss_average/epoch_step}')
                BER = test(model=model, selected_channels=selected_features)
                BERS_rep.append(BER)
            final_loss = loss_average/epoch_step
            epoch_step = 0
            loss_average = 0
    
    # create average BER measurements
    print(BERS_rep)
    BERS_AV.append((np.sum(BERS_rep)/REPETITION))
    print("BER averaged: ", BERS_AV[-1])
        
print(BERS_AV)

# # Save the checkpoint
# checkpoint = {
#     'epoch': epoch,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }

# torch.save(checkpoint, 'checkpoint.pth')

# print(losses)
# fig, axes = plt.subplots(ncols=1, nrows=2)
# axes[0].plot(np.linspace(1, num_epochs, num_epochs), BERS)
# axes[0].set_title('BER per epoch')
# axes[0].set_xlabel('Epochs')
# axes[0].set_ylabel('BER')
# axes[1].plot(np.linspace(1, num_epochs, num_epochs), losses)
# axes[1].set_title('Loss per epoch')
# axes[1].set_xlabel('Epochs')
# axes[1].set_ylabel('Loss')
# plt.show()
