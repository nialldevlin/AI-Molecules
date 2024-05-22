import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(GraphDecoder, self).__init__()
        self.conv1 = GCNConv(latent_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        # Using sigmoid for the final layer if the output is binary
        x = torch.sigmoid(self.conv2(x, edge_index))
        return x


class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(GraphVAE, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, output_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, log_var = self.encoder(data.x, data.edge_index)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z, data.edge_index), mu, log_var
