#!/home/ndev/miniconda3/bin/python

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np

from load_data import DataHandler, MoleculeDataset
from cnn_rnn import CNNtoRNN
# from vae import VAE
from model_handler import ModelHandler

# Model Parameters
channels = 3
cnn_output_size = 256
cnn_hidden_size = 128
lstm_hidden_size = 1024
num_lstm_layers = 2
num_cnn_layers = 4

# Data parameters
sequence_length = 20
batch_size = 16
train_split = 0.8

# Training parameters
learning_rate = 0.0001
epochs = 20
patience = 5

model_save_path = "lstm_molecule.pth"

train = True  # Set false to predict without training

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# filename = 'ti-.004cu_2500k_0.0/XDATCAR'
filename = 'current_train/XDATCAR'

data = DataHandler(filename, train_split)

train_dataset = MoleculeDataset(data.train_data, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MoleculeDataset(data.test_data, sequence_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("Creating model")
model = CNNtoRNN(channels=channels,
                 cnn_output_size=cnn_output_size,
                 cnn_hidden_size=cnn_hidden_size,
                 lstm_hidden_size=lstm_hidden_size,
                 num_lstm_layers=num_lstm_layers,
                 num_cnn_layers=num_cnn_layers)

# model = VAE(device, 750)
model.to(device)  # Move the model to GPU

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()  # Assuming a regression problem

# Train
handler = ModelHandler(model, device, optimizer,
                       loss_fn, train_loader, test_loader)

print("Training")
if train:
    # handler.train_early_stop(patience)
    handler.train_epochs(epochs)
    handler.eval()
    handler.save(model_save_path)
else:
    handler.load(model_save_path)

actual = []
predictions = []

print("Predicting")
predicted = 0
errors = []
for feature, target in test_dataset:
    actual.append(target)
    new = handler.generate_next(feature)
    loss = mean_squared_error(target, new)
    errors.append(loss)
    # print(f"Loss at prediction {predicted}: {loss}")
    predictions.append(new)
    predicted += 1
plt.plot(errors)
plt.show()

actual = np.array(actual)
predictions = np.array(predictions)

actual_reshaped = data.unNormalizeFeatures(actual)
predictions_reshaped = data.unNormalizeFeatures(predictions)

# Save as an xdatcar for visual validation
data.toXDATCAR(actual_reshaped, 'actual/XDATCAR')
data.toXDATCAR(predictions_reshaped, 'predicted/XDATCAR')

# errors = []
# for i in range(len(test_dataset)):
#     input, actual = test_dataset[i]
#     pred = generate_next(input)
#     error = mean_squared_error(pred, actual)
#     print(f"Prediction {i}, error {error}")
#     errors.append(error)

# plt.plot(errors)
# plt.show()
