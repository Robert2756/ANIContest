# Download der Daten
import os
import gdown
import time
import numpy as np

f = 'ANI-Contest.zip'
url = 'https://drive.google.com/uc?id=1AxA85q51-LI3uHE3IWVyX7B4R2ArIoPe'
if not os.path.exists(f):
    gdown.download(url, f, quiet=True)

TRAIN = 2800
TEST = 3250

features = [225, 8, 181, 180, 179, 178, 177, 176, 175]
# features = [i for i in range(0,250)]

# Zip-Datei entpacken
import zipfile
with zipfile.ZipFile(f, 'r') as zip_file:
    zip_file.extractall('.')

# Daten mittels Pandas in NumPy laden
import pandas as pd
training_input = pd.read_csv('ANI_Training.Input', header=None)
training_label = pd.read_csv('ANI_Training.Label', header=None)
test_input = pd.read_csv('ANI_Test.Input', header=None)

channels_train = training_input.iloc[0:TRAIN, features]
train_input = channels_train.to_numpy()

target_train = training_label.iloc[0:TRAIN]
target_train = target_train.to_numpy()

channels_test = training_input.iloc[TRAIN:TEST, features]
test_input = channels_test.to_numpy()

target_test = training_label.iloc[TRAIN:TEST]
target_test = target_test.to_numpy()

print("training_input: ", train_input.shape)
print("training_label: ", target_train.shape)

print("training_input: ", test_input.shape)
print("training_label: ", target_test.shape)
# print("test_input: ", test_input)

# Normalize each feature separately
min_vals = np.min(train_input, axis=0, keepdims=True)
max_vals = np.max(train_input, axis=0, keepdims=True)
print("min vals: ", min_vals)
print("min vals: ", max_vals)

train_input = 2 * (train_input - min_vals) / (max_vals - min_vals) - 1
print(train_input)



# Classificator
from tensorflow import keras
import keras
from keras import layers
import matplotlib.pyplot as plt

model = keras.models.Sequential()

# Input-Dimensionen festlegen (wir benutzen 2-dimensionale Eingaben)
input_dim = len(features)

# Eine Hiddenschicht hinzufügen (2 Neuronen)
model.add(keras.layers.Dense(512, input_dim=input_dim, activation='tanh'))
model.add(keras.layers.Dense(128, input_dim=input_dim, activation='tanh'))
model.add(keras.layers.Dense(64, input_dim=input_dim, activation='tanh'))
model.add(keras.layers.Dense(16, input_dim=input_dim, activation='tanh'))

# Eine Outputschicht hinzufügen (1 Neuron) mit Sigmoid-Aktivierung,
# um Ausgabe zwischen 0 und 1 zu erhalten
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(train_input, target_train, epochs=50,
                    batch_size=8)

def plot_training_progress(history):
    """ Trainingsfortschritt visualisieren

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Rückgabewert der Fit-Funktion.
    """
    # zu zeichnende Daten extrahieren
    hist = history.history
    epochs = history.epoch

    # Verlauf der Accuracy über die Epochen als rote Linie
    plt.plot(epochs, hist['accuracy'], 'r-')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    # Verlauf des Trainingsfehlers (Mean Squared Error) über die Epochen
    # als blaue Linie
    plt.plot(epochs, hist['loss'], 'b-')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()

plot_training_progress(history)

# Prädiktion auf unbekannten Daten
predictions = model.predict(test_input)



# validate

PREDICTIONS = []
# discretize predictions to [0,1]
for i, prediction in enumerate(predictions):
    if prediction>=0.5:
        PREDICTIONS.append(1)
    elif prediction<0.5:
        PREDICTIONS.append(0)

n = np.sum(np.array(target_test) == 0)
p = np.sum(np.array(target_test) == 1)

tp = 0
tn = 0
fp = 0
fn = 0

for i, pred in enumerate(PREDICTIONS):
    trgt = target_test[i]
    print("pred: ", pred)
    print("trgt: ", trgt)

    if pred == trgt and pred==1:
        tp += 1
    elif pred == trgt and pred==0:
        tn += 1
    elif pred != trgt and pred==1:
        fp += 1
    elif pred != trgt and pred==0:
        fn += 1

BER = 1/2*(int(fn)/int(p) + int(fp)/int(n))
print("BER on test dataset: ", BER)