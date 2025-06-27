import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define KL loss function
def kl_loss(z_mu, z_log_var):
    kl = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
    return kl

# Define reconstruction loss function
def recon_loss(data_orig, data_reconstructed):
    reconstruction_loss = F.mse_loss(data_reconstructed, data_orig, reduction='mean')
    return reconstruction_loss

# Define VAE loss function
def vae_loss(data_orig, data_reconstructed, z_mu, z_log_var):
    reconstruction_loss = recon_loss(data_orig, data_reconstructed)
    kl = kl_loss(z_mu, z_log_var)
    return reconstruction_loss + kl

# Define a basic VAE model with convolutional layers
class ConvBlockSeqRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlockSeqRes, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class VAE(nn.Module):
    def __init__(self, input_dim=22, start_filter_num=32, kernel_size=3, latent_dim=16, output_length=336):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            ConvBlockSeqRes(input_dim, start_filter_num, kernel_size, 1, 1),
            nn.MaxPool1d(2),
            ConvBlockSeqRes(start_filter_num, int(start_filter_num * 1.5), kernel_size, 1, 1),
            nn.MaxPool1d(2),
            ConvBlockSeqRes(int(start_filter_num * 1.5), start_filter_num * 3, kernel_size, 1, 1),
            nn.MaxPool1d(2),
            ConvBlockSeqRes(start_filter_num * 3, int(start_filter_num * 4.5), kernel_size, 1, 1),
            nn.MaxPool1d(2),
            ConvBlockSeqRes(int(start_filter_num * 4.5), start_filter_num * 6, kernel_size, 1, 1),
            nn.MaxPool1d(2),
            ConvBlockSeqRes(start_filter_num * 6, int(start_filter_num * 7.5), kernel_size, 1, 1),
            nn.MaxPool1d(2),
            ConvBlockSeqRes(int(start_filter_num * 7.5), start_filter_num * 9, kernel_size, 1, 1),
            nn.Flatten()
        )

        # Latent space
        self.fc_mu = nn.Linear(1440, latent_dim)  # Adjust input size accordingly
        self.fc_log_var = nn.Linear(1440, latent_dim)

        # Decoder
        self.start_filter_num = start_filter_num
        self.input_dim = input_dim
        self.output_length = output_length
        self.decoder_fc = nn.Linear(latent_dim, start_filter_num * 9)  # Adjust output size accordingly
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(start_filter_num * 9, start_filter_num * 7, kernel_size, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(start_filter_num * 7, start_filter_num * 5, kernel_size, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose1d(start_filter_num * 5, start_filter_num * 3, kernel_size, 1, 1),
            # nn.ReLU(),
            # nn.ConvTranspose1d(start_filter_num * 3, 1, kernel_size, 1, 1),
            nn.Sigmoid()
        )
        self.predictor = nn.Linear(start_filter_num * 3, input_dim * output_length)

    def encode(self, x):
        h = self.encoder(x.transpose(1,2))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, self.start_filter_num*9, 1)  # Reshape based on the size before Flatten
        h = self.decoder(h)
        return self.predictor(h.squeeze()).reshape(-1, self.output_length, self.input_dim)

    def forward(self, x):
        self.encode(x)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var
