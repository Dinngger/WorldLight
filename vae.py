import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class VAE(nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU()
        )
        self.enc_mu = nn.Linear(H, latent_size)
        self.enc_log_sigma = nn.Linear(H, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, H),
            nn.ReLU(),
            nn.Linear(H, D_in),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        return mu, log_sigma

    def forward(self, state):
        mu, log_sigma = self.encode(state)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return self.decoder(eps * std + mu), mu, log_sigma


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      # Normalize the images to be -0.5, 0.5
#      transforms.Normalize(0.5, 1)]
#     )
mnist = torchvision.datasets.MNIST('/media/dinger/inner/Dataset/pytorch_data', train=True, download=False, transform=transforms.ToTensor())

input_dim = 28 * 28
batch_size = 128
num_epochs = 30
hidden_size = 512
latent_size = 11

device = torch.device('cuda')
print('Number of samples: ', len(mnist))

dataloader = torch.utils.data.DataLoader(
    mnist, batch_size=batch_size,
    shuffle=True,
    pin_memory=True)
inputs = list(map(lambda x: x[0].to(device, non_blocking=True).view(-1, input_dim), list(dataloader)))

vae = VAE(input_dim, hidden_size, latent_size).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
for epoch in range(num_epochs):
    for x in inputs:
        optimizer.zero_grad()
        p_x, mu, log_sigma = vae(x)
        resons_loss = F.mse_loss(p_x, x)
        kl = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim=1), dim=0)
        loss = resons_loss + kl * 0.003
        loss.backward()
        optimizer.step()
    print(epoch, loss.item(), resons_loss.detach().item(), kl.detach().item())

z = torch.randn(batch_size, latent_size).to(device)
out = vae.decoder(z).view(-1, 1, 28, 28)
torchvision.utils.save_image(out, 'vae_sampled.png')

out, _, _ = vae(inputs[0])
x_concat = torch.cat([inputs[0].view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
torchvision.utils.save_image(x_concat, 'vae_reconst.png')
