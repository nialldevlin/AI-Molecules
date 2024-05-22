#!/home/ndev/miniconda3/bin/python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

# from linear_vae import LinearVAE
from conv_vae import ConvVAE

from load_data import *

# Data parameters
batch_size = 8
train_split = 1
dim = 750  # TODO calculate this

# model parameters
hidden_dim = 512
latent_dim = 256

# Training parameters
learning_rate = 0.0001
epochs = 100

# model_save_path = "linear_vae.pth"
model_save_path = "conv_vae.pth"

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data")

# Replace the path with the relative path to your data
current_dir = os.path.dirname(__file__)
filename = os.path.join(current_dir, '../../data/ti-.004cu_2500k_0.0/XDATCAR')

data, header = loadXDATCAR(filename)

train_dataset = LinearVAEMoleculeDataset(data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 750 is the number of data points in a single feature, in this case 1 timestep
print("Creating model")
# model = LinearVAE(device, 750)
model = ConvVAE(device, 750)
model.to(device)  # Move the model to GPU

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def loss_function(x, x_hat, mean, log_var):
    # reproduction_loss = nn.functional.binary_cross_entropy(
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train(model, optimizer, epochs, device):
    model.train()
    try:
        for epoch in range(epochs):
            overall_loss = 0
            num_loss = 0
            for batch_idx, x in enumerate(train_loader):
                x = x.float()
                # cbs = x.size(0)
                # x = x.view(cbs, x_dim).to(device)
                x = x.to(device)

                optimizer.zero_grad()

                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)

                overall_loss += loss.item()
                num_loss += 1

                loss.backward()
                optimizer.step()

            print("\tEpoch", epoch + 1, "\tAverage Loss: ",
                  overall_loss/num_loss)
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
    finally:
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to", model_save_path)
    return overall_loss


print("Training")
train(model, optimizer, epochs=epochs, device=device)

# model.load_state_dict(torch.load(model_save_path))
# model.to(device)

model.eval()

# Uncomment this to validate on existing data

# print("Predicting")
# actual_sample = train_dataset[0]
# actual_tensor = torch.tensor(
#     actual_sample, dtype=torch.float32).unsqueeze(0).to(device)
# with torch.no_grad():
#     predicted_tensor, mean, logvar = model(actual_tensor)
#     predicted_sample = predicted_tensor.squeeze().cpu().numpy()

# print("Saving to XDATCAR")
# predicted_normalized = train_dataset.unNormalizeFeatures(predicted_sample)
# actual_normalized = train_dataset.unNormalizeFeatures(actual_sample)

# if not os.path.exists('predicted'):
#     os.makedirs('predicted')
# if not os.path.exists('actual'):
#     os.makedirs('actual')

# toXDATCAR("predicted/XDATCAR", np.array(predicted_normalized), header)
# toXDATCAR("actual/XDATCAR", np.array(actual_normalized), header)


print('Generating')


def generate(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    return x_decoded.detach().cpu()


generated_sample = generate(0.0, 1.0)

generated_normalized = train_dataset.unNormalizeFeatures(generated_sample)

if not os.path.exists('generated'):
    os.makedirs('generated')

toXDATCAR("generated/XDATCAR", np.array(generated_normalized), header)
