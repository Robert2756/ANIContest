import time
import torch
import numpy as np
import pandas as pd

from learner import MLP
from learner import ContestDataset

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

TH = 0.5
path_input = "./data/ANI_Training.Input"
path_target = "./data/ANI_Training.Label"

selected_features = [225, 156, 132, 125, 58, 151, 172, 120, 117]

dataset = ContestDataset(selected_channels=selected_features)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Number of classifiers
n_classifiers = 5

# Initialize weights
sample_weights = [np.ones(len(dataset)) / len(dataset)]
weights = []

# Store classifiers and their alpha values
classifiers = []
alphas = []
num_epochs = 1

# Loss and optimizer
criterion = nn.MSELoss()  # Example loss function (Mean Squared Error)
# criterion = nn.BCELoss()  # Binary Cross Entropy Loss

for i in range(n_classifiers):
    # Create and train the MLP
    model = MLP(input_size=len(selected_features), hidden_size=256, output_size=1).to("cpu")

    # Learning rate scheduler and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    step = 0

    # Load the checkpoint
    ckpt = 'checkpoint' + str(i+1) + '.pth'
    checkpoint = torch.load(ckpt)

    # Load the state_dict of the model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # inference
    model.eval()

    PREDICTIONS = []

    df = pd.read_csv(path_input, header=None) # (3250, 250)
    df_test = df.iloc[2900:3250, :]

    target_train = pd.read_csv(path_target, header=None) # (3250, 1)
    target_train = target_train.iloc[2900:3250, :]
    target = target_train

    data = np.array(df_test.T) # (250,  3250)
    data = data[selected_features, ::]

    # Normalize each feature separately
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    # print("min vals: ", min_vals)
    # print("min vals: ", max_vals)

    data = 2 * (data - min_vals) / (max_vals - min_vals) - 1

    # iterate over test data
    for i in range(np.array(data).shape[1]):
        output = model(torch.tensor(data[:,i], dtype=torch.float32))

        # make outputs discrete
        if output>=0.5:
            PREDICTIONS.append(1)
        elif output<0.5:
            PREDICTIONS.append(0)
    
    # Calculate error
    TARGET = list(np.array(target).squeeze(-1))

    incorrect = [i for i, val in enumerate(PREDICTIONS) if val != TARGET[i]]
    print("incorrect: ", incorrect)
    error = np.sum(sample_weights[-1][incorrect])
    print("error: ", error)
    weight = 1/2 * np.log((1-error)/error) # weight for weak learner
    # print("weight: ", weight)
    false_weights = np.power(np.e, -weight)
    true_weights = np.power(np.e, weight)

    weights_unnormalized = []
    for index, weight_val in enumerate(sample_weights[-1]):
        if index in incorrect:
            weights_unnormalized.append(weight_val*false_weights)
        else:
            weights_unnormalized.append(weight_val*true_weights)
    
    Z = np.sum(weights_unnormalized)

    print("sample weights: ", sample_weights[-1][incorrect])
    sample_weights.append(weights_unnormalized/Z)
    weights.append(weight)
    print("New sample weights: ", np.array(sample_weights).shape)
    time.sleep(10)

print("Weights: ", weights)
print("Weights shape: ", np.array(weights).shape)

# es werden von jedem checkpoint immer die gleichen samples falsch klassifiziert, was bedeutet dass die KLassifikatoren nicht divers genug sind um zu einem strong learner kombiniert zu werden
# die propabilities der flasch klassifizierten Datenpunkte sinken rapide mit jeder Runde