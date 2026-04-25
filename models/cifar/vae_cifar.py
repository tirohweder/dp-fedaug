import torch
import torch.nn as nn


class CIFAR_VAE(nn.Module):
    """
    VAE for CIFAR-10 with GroupNorm (DP-SGD compatible).
    Input: 32x32x3 RGB images.
    """
    def __init__(self, num_latent_dims=64, num_img_channels=3, img_size=32):
        super().__init__()
        self.latent_dim = num_latent_dims

        self.encoder = nn.Sequential(
            nn.Conv2d(num_img_channels, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, num_latent_dims)
        self.fc_logvar = nn.Linear(128 * 4 * 4, num_latent_dims)

        self.fc_decode = nn.Linear(num_latent_dims, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, num_img_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.kl_div = 0.0

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        self.kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.decode(z)
