import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from load_poscar import *
from graph_vae_2 import *
import numpy as np
import random

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

k = 12
dir = 'MG_data'
mf = MoleculeFrames(dir, k)

loader = DataLoader(mf.knn_data, batch_size=16, shuffle=True)
num_features = mf.knn_data[0].num_features

model = GraphVAE(num_features, 512)
optimizer = Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data.x, data.edge_index)
        loss = loss_function(recon_batch, data.x, mu, log_var)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader.dataset)


for epoch in range(0, 50):
    loss = train()
    print(f'Epoch: {epoch}, Loss: {loss:.4f}')


def test(data):
    model.eval()
    with torch.no_grad():
        # Forward pass through the model
        recon_data, mu, log = model(data.x, data.edge_index)
        loss = loss_function(recon_data, data.x, mu, log)
        return recon_data, loss


# Re predict from training data
i = random.randint(0, len(mf.knn_data))
# o = mf.knn_data[i].x.squeeze().numpy()
sample, loss = test(mf.knn_data[i])
print(loss)
sample = sample.squeeze().numpy()
# sample = dataset.unNormalizeFeature(sample)
# o = dataset.unNormalizeFeature(o)
print(sample.shape)
mf.toPOSCAR("vae/Predicted/POSCAR", mf.headers[i], sample)
mf.toPOSCAR("vae/Actual/POSCAR", mf.headers[i], mf.data[i])

# model.eval()
# with torch.no_grad():
#     generated = model.generate()
#     generated = generated.squeeze().numpy()
#     print(generated)
#     # generated = dataset.unNormalizeFeature(generated)
#     mf.toPOSCAR("vae/Predicted/XDATCAR", np.array([generated]))
