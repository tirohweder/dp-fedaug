"""
vae_braintumor.py - DP-Compliant VAE (GroupNorm)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_latent_dims, num_img_channels, max_num_filters, device, img_size=64):
        super().__init__()
        self.device = device

        # Filters: 32 -> 64 -> 128
        f1, f2, f3 = max_num_filters // 4, max_num_filters // 2, max_num_filters

        self.conv1 = nn.Conv2d(num_img_channels, f1, 3, 2, 1)
        self.conv2 = nn.Conv2d(f1, f2, 3, 2, 1)
        self.conv3 = nn.Conv2d(f2, f3, 3, 2, 1)

        # [DP-CHANGE] GroupNorm instead of BatchNorm
        # 8 groups is a standard heuristic that works well for these channel counts
        self.gn1 = nn.GroupNorm(8, f1)
        self.gn2 = nn.GroupNorm(8, f2)
        self.gn3 = nn.GroupNorm(8, f3)

        flattened_dim = f3 * (img_size // 8) * (img_size // 8)
        self.proj_mu = nn.Linear(flattened_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(flattened_dim, num_latent_dims)
        self.kl_div = 0

    def forward(self, x):
        x = F.leaky_relu(self.gn1(self.conv1(x)))
        x = F.leaky_relu(self.gn2(self.conv2(x)))
        x = F.leaky_relu(self.gn3(self.conv3(x)))
        x = torch.flatten(x, 1)

        mu = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        sigma = torch.exp(0.5 * logvar)
        z = sigma * torch.randn_like(sigma) + mu

        # Store KL for loss computation
        self.kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z


class Decoder(nn.Module):
    def __init__(self, num_latent_dims, num_img_channels, max_num_filters, img_size=64):
        super().__init__()
        f1 = max_num_filters
        f2 = max_num_filters // 2
        f3 = max_num_filters // 4

        self.input_shape = [f1, img_size // 8, img_size // 8]
        flattened_dim = f1 * (img_size // 8) * (img_size // 8)

        self.lin1 = nn.Linear(num_latent_dims, flattened_dim)

        self.conv1 = nn.ConvTranspose2d(f1, f2, 3, 2, 1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(f2, f3, 3, 2, 1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(f3, num_img_channels, 3, 2, 1, output_padding=1)

        # [DP-CHANGE] GroupNorm instead of BatchNorm
        self.gn1 = nn.GroupNorm(8, f2)
        self.gn2 = nn.GroupNorm(8, f3)

    def forward(self, z):
        x = self.lin1(z)
        x = x.view(-1, *self.input_shape)
        x = F.leaky_relu(self.gn1(self.conv1(x)))
        x = F.leaky_relu(self.gn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x


class VAE(nn.Module):
    def __init__(self, num_latent_dims, num_img_channels, max_num_filters, device, img_size=64):
        super().__init__()
        self.encoder = Encoder(num_latent_dims, num_img_channels, max_num_filters, device, img_size)
        self.decoder = Decoder(num_latent_dims, num_img_channels, max_num_filters, img_size)
        self.kl_div = 0

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        self.kl_div = self.encoder.kl_div
        return x_recon

    def decode(self, z):
        return self.decoder(z)