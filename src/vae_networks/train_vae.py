#!/home/ndev/miniconda3/bin/python

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from load_data import DataHandler, VAEMoleculeDataset
from conv_vae import ConvVAE

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

model_save_path = "conv_vae.pth"

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data")

filename = 'current_train/XDATCAR'

data = DataHandler(filename, train_split, flatten=True, transpose=False)

train_dataset = VAEMoleculeDataset(data.train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Creating model")
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

print("Predicting")
x = train_dataset[0]
xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    xp, mean, logvar = model(xt)
    xn = xp.squeeze().cpu().numpy()

print("Saving to XDATCAR")
xnr = data.unNormalizeFeature(xn)
xr = data.unNormalizeFeature(x)
data.toXDATCAR(np.array(xnr), "vae/Predicted/XDATCAR")
data.toXDATCAR(np.array(xr), "vae/Actual/XDATCAR")


# def generate(mean, var):
#     z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
#     x_decoded = model.decode(z_sample)
#     return x_decoded.detach().cpu()


# nf = generate(0.0, 1.0)

# nfr = data.unNormalizeFeatures(nf)
# data.toXDATCAR(nfr, "vae/XDATCAR")
