import time
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim

from learner import MLP

TH = 0.5
path_input = "./data/ANI_Training.Input"
# path_input = "/home/viehrt/Documents/ANiContest/ANIContest/data/ANI_Training.Input"
path_target = "./data/ANI_Training.Label"
# path_target = "/home/viehrt/Documents/ANiContest/ANIContest/data/ANI_Training.Label"

def test(model, selected_channels):
    PREDICTIONS = []

    df = pd.read_csv(path_input, header=None) # (3250, 250)
    df_test = df.iloc[2500:3250, :]

    target_train = pd.read_csv(path_target, header=None) # (3250, 1)
    target_train = target_train.iloc[2500:3250, :]
    target = target_train

    data = np.array(df_test.T) # (250,  3250)
    data = data[selected_channels, ::]

    # Normalize each feature separately
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    # print("min vals: ", min_vals)
    # print("min vals: ", max_vals)

    data = 2 * (data - min_vals) / (max_vals - min_vals) - 1

    # iterate over test data
    for i in range(np.array(data).shape[1]):
        output = model(torch.tensor(data[:,i], dtype=torch.float32))
        # print("output: ", output)
        # print("target: ", np.array(target)[i])
        # make outputs discrete
        if output>=0.5:
            PREDICTIONS.append(1)
        elif output<0.5:
            PREDICTIONS.append(0)
            
    
    # compare with labels
    predictions = np.array(PREDICTIONS).reshape(-1, 1)

    n = np.sum(np.array(target) == 0)
    p = np.sum(np.array(target) == 1)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i, pred in enumerate(predictions):
        trgt = np.array(target).squeeze()[i]
        pred = pred[0]
        # print("pred: ", pred)
        # print("trgt: ", trgt)

        if pred == trgt and pred==1:
            tp += 1
        elif pred == trgt and pred==0:
            tn += 1
        elif pred != trgt and pred==1:
            fp += 1
        elif pred != trgt and pred==0:
            fn += 1

    BER = 1/2*(int(fn)/int(p) + int(fp)/int(n))
    print(tp, tn, fp, fn)
    print("BER on test dataset: ", BER)
    return BER


# selected_channels = [225, 156, 132, 125, 58, 151, 172, 120, 117]
# model = MLP(input_size=len(selected_channels), hidden_size=512, output_size=1).to("cpu")
# optimizer = optim.Adam(model.parameters(), lr=0.005)

# # Load the checkpoint
# checkpoint = torch.load('checkpoint_BER_0.25705.pth')

# # Load the state_dict of the model and optimizer
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# # inference
# model.eval()

# test(model=model, selected_channels=selected_channels)

