import torch
import torch.nn as nn


class ConvVAE(nn.Module):

    def __init__(self, device, input_dim):
        super(ConvVAE, self).__init__()

        self.device = device

        self.d1 = 512
        self.d2 = 512
        self.d3 = 256

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.d1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.d1, self.d2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.d2, self.d3),
            nn.LeakyReLU(0.2)
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(self.d3, 2)
        self.logvar_layer = nn.Linear(self.d3, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, self.d3),
            nn.LeakyReLU(0.2),
            nn.Linear(self.d3, self.d2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.d2, self.d1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.d1, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    # def forward(self, x):
    #     mean, logvar = self.encode(x)
    #     z = self.reparameterization(mean, logvar)
    #     x_hat = self.decode(z)
    #     return x_hat, mean, logvar

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var
