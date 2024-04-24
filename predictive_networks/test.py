#!/home/ndev/miniconda3/bin/python

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from load_data import DataHandler, MoleculeDataset
from lstm_model import LSTMModel
from model_handler import ModelHandler

hidden_size = 512  # Can be tuned
num_layers = 3  # Number of LSTM layers
sequence_length = 50  # Number of previous sequences to consider
batch_size = 16
learning_rate = 0.001
train_split = 0.8
num_epochs = 20
model_save_path = "lstm_molecule.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

filename = '../test_xdatcars/XDATCAR_1'

data = DataHandler(filename, train_split)

train_dataset = MoleculeDataset(data.train_data, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MoleculeDataset(data.test_data, sequence_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

input_size = len(train_dataset.data[0])

model = LSTMModel(input_size, hidden_size, num_layers)  # Assuming LSTMModel is your model class
model.to(device)  # Move the model to GPU

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()  # Assuming a regression problem

handler = ModelHandler(model, device, optimizer, loss_fn, train_loader, test_loader)

handler.load(model_save_path)

actual = []
predictions = []

predicted = 0
for feature, target in test_dataset:
    # actual.append(target)
    new = handler.generate_next(feature)
    # loss = mean_squared_error(target, new)
    # print(f"Loss at prediction {predicted}: {loss}")
    predictions.append(new)
    # predicted += 1

predictions = np.array(predictions)
# predictions_reshaped = data.unNormalizeFeatures(predictions)
# print(predictions_reshaped[1] - predictions_reshaped[0])
# data.toXDATCAR(predictions_reshaped, 'predicted/XDATCAR')

# f1, t1 = test_dataset[30]
# f2, t2 = test_dataset[45]

# p1 = handler.generate_next(f1)
# p2 = handler.generate_next(f2)

# print(p1 - p2)
