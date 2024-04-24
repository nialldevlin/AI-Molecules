from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
# Ensure this module is prepared to handle 3D inputs
from graph_vae_2 import GraphVAE
from load_poscar import MoleculeFrames


class GraphDataset(Dataset):
    def __init__(self, numpy_arrays, num_neighbors=12):
        self.graphs = [self.numpy_to_graph(
            arr, num_neighbors) for arr in numpy_arrays]

    def numpy_to_graph(self, array, num_neighbors):
        # Convert XYZ coordinates to a graph using N nearest neighbors
        n_atoms = array.shape[0]
        x = torch.tensor(array, dtype=torch.float)
        tree = cKDTree(array)
        src, dst = [], []

        # Find N nearest neighbors for each atom
        for i in range(n_atoms):
            # +1 because query includes self
            distances, indices = tree.query(array[i], k=num_neighbors+1)
            for j in indices:
                if i != j:  # Avoid self-loop
                    src.append(i)
                    dst.append(j)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def train(model, data_loader, optimizer, epochs=20):
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            reconstruction, mu, log_var = model(data)
            loss_reconstruction = criterion(reconstruction, data.x)
            loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = loss_reconstruction + loss_kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")


# Example usage
mf = MoleculeFrames('MG_data', 12)
dataset = GraphDataset(mf.data, num_neighbors=12)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

model = GraphVAE(input_dim=3, hidden_dim=32, latent_dim=16, output_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, data_loader, optimizer)


def predict(model, graph_data):
    model.eval()
    with torch.no_grad():
        reconstruction, _, _ = model(graph_data)
        return reconstruction


# Predict a sample from the dataset
sample_graph = dataset[0]
predicted_graph = predict(model, sample_graph)
print("Original Node Features (XYZ Coordinates):", sample_graph.x)
print("Predicted Node Features (XYZ Coordinates):", predicted_graph)


def visualize_graph(graph_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract node positions
    xyz = graph_data.numpy()
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Plot nodes
    ax.scatter(x, y, z, c='r', label='Atoms')

    # Extract edges
    # edge_index = graph_data.edge_index.numpy()
    # for i in range(edge_index.shape[1]):
    #     start, end = edge_index[:, i]
    #     ax.plot([x[start], x[end]], [y[start], y[end]],
    #             [z[start], z[end]], 'gray')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Visualization of Graph Data')
    ax.legend()
    plt.show()

visualize_graph(sample_graph.x)
visualize_graph(predicted_graph)
