# Improvements

## tester
train dataaset: 
df_train = df.iloc[0:2800, :]

test dataset
df_test = df.iloc[2800:3250, :]

-> use all data for final training

## Selected channels with MIFS algorithm
Selected features:  [225, 182, 8, 166, 165, 164, 4, 220, 219, 86, 161, 160, 159, 158, 157, 156, 155, 154]

BER on test dataset, hidden_size=64, 3 dense layers, 5 input features (trained on full dataset)
BER: 0.21681

BER on test dataset, hidden_size=128, 3 Dense layers, 7 input features (trained only on train dataset)
BER: 0.30256400087251384; checkpoint1

BER on test dataset, hidden_size=128, 4 Dense layers, 7 input features (trained only on train dataset)
BER: 0.3371472763687561; checkpoint2

BER on test dataset, hidden_size=512, 3 Dense layers, 7 input features (trained only on train dataset)
BER:  0.32570544726248785

BER on test dataset, hidden_size=512, 3 Dense layers, 7 input features (trained only on train dataset), lr decay of 0.8, 250 epochs
BER:  0.32822383945745504; checkpoint 3

BER on test dataset, hidden_size=512, 3 Dense layers, 10 input features (trained only on train dataset), lr decay of 0.8, 250 epochs
BER:  0.32152134684407785; checkpoint 4

BER on test dataset, hidden_size=512, 3 Dense layers, 18 input features (trained only on train dataset), lr decay of 0.8 every 20000 steps, 250 epochs
BER:  0.3042297090959567; checkpoint 5

BER on test dataset, hidden_size=512, 3 Dense layers, 18 input features (trained on whole dataset), lr decay of 0.8 every 20000 steps, 250 epochs
BER:  ; checkpoint_final1 
Loss of final epoch: 0.006756756920367479

BER on test dataset, hidden_size=512, 3 Dense layers, 18 input features (trained on whole dataset), lr decay of 0.8 every 10000 steps, 250 epochs
BER:  ; checkpoint_final2 
Loss of final epoch: 0.008599508553743362

BER on test dataset, hidden_size=512, 3 Dense layers, 18 input features (trained on whole dataset), lr decay of 0.8 every 30000 steps, 250 epochs
BER:  ; checkpoint_final3
Loss of final epoch: 0.0070638819597661495

# Overfitting
## Adjusting MIFS algorithm
beta = 0.1 (changed from 0.5)

[225] -> 0.12657137027150012
[8] -> 0.043420045275251444
[181] -> 0.03613711693954798
[180] -> 0.03557478345119342
[179] -> 0.02905135698370739
[178] -> 0.023041340694343403
[177] -> 0.017362894473898507
[176] -> 0.011365264752842198
[175] -> 0.005567026051854912
[174] -> -0.0008536679528394139

selected features: [225, 8, 181, 180, 179, 178, 177, 176, 175]