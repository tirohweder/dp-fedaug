"""
Variational Autoencoder (VAE) for Brain Tumor Dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    A convolutional variational encoder that adapts to different image sizes.
    The input image shape is (num_img_channels, img_size, img_size), and after 3
    stride-2 convolutional layers the spatial dimensions become (img_size // 8, img_size // 8).
    """
    def __init__(self, num_latent_dims, num_img_channels, max_num_filters, device, img_size=64):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.device = device

        # Define input shape based on the parameter
        img_input_shape = (num_img_channels, img_size, img_size)

        # Define filter counts
        num_filters_1 = max_num_filters // 4
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters

        # Convolutional layers (stride=2 halves spatial dims each time)
        self.conv1 = nn.Conv2d(num_img_channels, num_filters_1, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=3, stride=2, padding=1)

        # Optional shortcut layers (if needed)
        self.shortcut2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=1, stride=2, padding=0)
        self.shortcut3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=1, stride=2, padding=0)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(num_filters_1)
        self.bn2 = nn.BatchNorm2d(num_filters_2)
        self.bn3 = nn.BatchNorm2d(num_filters_3)

        # After 3 conv layers with stride 2, the spatial dims become: img_size // 8 x img_size // 8
        flattened_dim = num_filters_3 * (img_size // 8) * (img_size // 8)
        self.proj_mu = nn.Linear(flattened_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(flattened_dim, num_latent_dims)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch dimension

        mu = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        sigma = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        # Compute KL divergence (for the loss later)
        self.kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return z


class Decoder(nn.Module):
    """
    A convolutional decoder that mirrors the encoder.
    It starts from a latent vector that is linearly mapped to a feature tensor
    with spatial dimensions (img_size // 8, img_size // 8) and upsamples it to produce
    an image of shape (num_img_channels, img_size, img_size).
    """
    def __init__(self, num_latent_dims, num_img_channels, max_num_filters, img_size=64):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters

        # Define output image shape
        img_output_shape = (num_img_channels, img_size, img_size)

        # Decoder layers configuration:
        # We'll use max_num_filters for the initial feature map, then progressively reduce
        num_filters_1 = max_num_filters       # e.g. 128 if max_num_filters is 128
        num_filters_2 = max_num_filters // 2    # e.g. 64
        num_filters_3 = max_num_filters // 4    # e.g. 32

        # The initial spatial dimensions are img_size // 8 x img_size // 8 (mirror the encoder)
        self.input_shape = [num_filters_1, img_size // 8, img_size // 8]
        flattened_dim = num_filters_1 * (img_size // 8) * (img_size // 8)

        # Linear layer maps the latent vector to the flattened feature space
        self.lin1 = nn.Linear(num_latent_dims, flattened_dim)

        # Transposed convolutions to upsample the feature map
        self.conv1 = nn.ConvTranspose2d(num_filters_1, num_filters_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(num_filters_2, num_filters_3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(num_filters_3, num_img_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Optional shortcut layers
        self.shortcut1 = nn.ConvTranspose2d(num_filters_1, num_filters_2, kernel_size=1, stride=2, padding=0, output_padding=1)
        self.shortcut2 = nn.ConvTranspose2d(num_filters_2, num_filters_3, kernel_size=1, stride=2, padding=0, output_padding=1)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(num_filters_2)
        self.bn2 = nn.BatchNorm2d(num_filters_3)

    def forward(self, z):
        x = self.lin1(z)
        x = x.view(-1, *self.input_shape)  # reshape to (batch, num_filters_1, img_size//8, img_size//8)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))  # output pixel values in [0, 1]
        return x


class VAE(nn.Module):
    """
    A convolutional Variational Autoencoder that supports different image sizes.
    It uses the Encoder and Decoder classes defined above.

    This implementation mirrors the convolutional VAE structure used in the
    rest of this repository.
    """
    def __init__(self, num_latent_dims, num_img_channels, max_num_filters, device, img_size=64):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters
        self.device = device

        self.encoder = Encoder(num_latent_dims, num_img_channels, max_num_filters, device, img_size=img_size)
        self.decoder = Decoder(num_latent_dims, num_img_channels, max_num_filters, img_size=img_size)
        self.kl_div = 0

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        self.kl_div = self.encoder.kl_div
        return x_recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location=self.device))
        self.eval()
