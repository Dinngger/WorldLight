import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class Invertible1x1Conv(nn.Module):
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        if reverse:
            if not hasattr(self, 'W_inverse'):
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


class VAE(nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, latent_size),
            nn.BatchNorm1d(latent_size, affine=False)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, H),
            nn.ReLU(),
            nn.Linear(H, D_in),
            nn.Sigmoid()
        )

    def forward(self, state):
        z = self.encoder(state)
        return self.decoder(z)


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      # Normalize the images to be -0.5, 0.5
#      transforms.Normalize(0.5, 1)]
#     )
mnist = torchvision.datasets.MNIST('/media/dinger/inner/Dataset/pytorch_data', train=True, download=False, transform=torchvision.transforms.ToTensor())

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
vae.train()
for epoch in range(num_epochs):
    for x in inputs:
        optimizer.zero_grad()
        p_x = vae(x)
        loss = F.mse_loss(p_x, x)
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())

vae.eval()

z = torch.randn(batch_size, latent_size).to(device)
out = vae.decoder(z).view(-1, 1, 28, 28)
torchvision.utils.save_image(out, 'bae_sampled.png')

out = vae(inputs[0])
x_concat = torch.cat([inputs[0].view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
torchvision.utils.save_image(x_concat, 'bae_reconst.png')
