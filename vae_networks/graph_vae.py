import torch
from torch_geometric.nn import GCNConv, Linear
from torch.nn import functional as F


class GraphVAE(torch.nn.Module):
    def __init__(self, num_features, d1):
        super(GraphVAE, self).__init__()
        # torch.manual_seed(random.randint(0, 1000000000))

        self.d1 = d1
        self.d2 = self.d1 // 2
        self.d3 = self.d2 // 2

        # Encoder
        self.conv1 = GCNConv(num_features, self.d1)
        self.conv2 = GCNConv(self.d1, self.d2)
        # Latent space
        self.mu = Linear(self.d2, self.d3)
        self.log_var = Linear(self.d2, self.d3)
        # Decoder
        self.conv3 = GCNConv(self.d3, self.d2)
        self.conv4 = GCNConv(self.d2, self.d1)
        self.conv5 = GCNConv(self.d1, num_features)

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        m = self.mu(h)
        l = self.log_var(h)
        return m, l

    # def reparameterize(self, mu, log_var):
    #     std = torch.exp(log_var / 2)
    #     eps = torch.randn_like(std)
    #     return eps.mul(std).add_(mu)

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + log_var*epsilon
        return z

    def decode(self, z, edge_index):
        h = F.relu(self.conv3(z, edge_index))
        h = F.relu(self.conv4(h, edge_index))
        # Using sigmoid for reconstruction
        return torch.sigmoid(self.conv5(h, edge_index))

    def forward(self, x, edge_index):
        mu, log_var = self.encode(x, edge_index)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, edge_index), mu, log_var

    def generate(self, num_samples=1):
        # Randomly generate mu and log_var
        mu = torch.randn(num_samples, self.d3)
        log_var = torch.randn(num_samples, self.d3)  # Assuming some variance

        # Reparameterize to sample z from the distribution defined by mu and log_var
        z = self.reparameterize(mu, log_var)

        # Generate a fully connected graph for num_samples nodes
        # Note: This might be computationally intensive for large num_samples
        rows, cols = torch.meshgrid(torch.arange(
            num_samples), torch.arange(num_samples), indexing='ij')
        edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)
        # Remove self-loops
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        generated_data = self.decode(z, edge_index)
        return generated_data


def loss_function(recon_x, x, mu, log_var):
    # Reconstruction loss (e.g., MSE for continuous data)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss
