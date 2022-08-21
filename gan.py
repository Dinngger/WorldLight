
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Generator(nn.Module):
    def __init__(self, latent_size, H, D_in):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, H),
            nn.ReLU(),
            nn.Linear(H, D_in),
            nn.Sigmoid()
        )
 
    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, D_in, H):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

    def forward(self, img):
        return self.model(img)


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      # Normalize the images to be -0.5, 0.5
#      transforms.Normalize(0.5, 1)]
#     )
mnist = torchvision.datasets.MNIST('/media/dinger/inner/Dataset/pytorch_data', train=True, download=False, transform=transforms.ToTensor())

input_dim = 28 * 28
batch_size = 128
num_epochs = 1000
hidden_size = 512
latent_size = 11

device = torch.device('cuda')
print('Number of samples: ', len(mnist))

dataloader = torch.utils.data.DataLoader(
    mnist, batch_size=batch_size,
    shuffle=True,
    pin_memory=True)
inputs = list(map(lambda x: x[0].to(device, non_blocking=True).view(-1, input_dim), list(dataloader)))

generator = Generator(latent_size, hidden_size, input_dim).to(device)
discriminator = Discriminator(input_dim, hidden_size).to(device)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)

fig, ax = plt.subplots()
ax.set_axis_off()
ld = []
lg = []

def update(n):
    for i, real_imgs in enumerate(inputs):
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_size).to(device)
        fake_imgs = generator(z).detach()
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        loss_D.backward()
        optimizer_D.step()
        for p in discriminator.parameters():
            p.data.clamp_(-1, 1)
        if i % 5 == 0:
            optimizer_G.zero_grad()
            gen_imgs = generator(z)
            loss_G = -torch.mean(discriminator(gen_imgs))
            loss_G.backward()
            optimizer_G.step()
    ld.append(loss_D.item())
    lg.append(loss_G.item())
    plt.clf()
    plt.subplot(121)
    plt.plot(ld, 'r')
    plt.plot(lg, 'b')
    plt.subplot(122)
    img = torchvision.utils.make_grid(gen_imgs.data[:128].view(-1, 1, 28, 28))
    img = np.squeeze(img.detach().cpu().numpy())
    img = np.transpose(img, [1,2,0])
    plt.imshow(img)

ani = FuncAnimation(fig, update, frames=num_epochs, repeat=False)  # 创建动画效果
plt.show()  # 显示图
# torchvision.utils.save_image(gen_imgs.data[:128].view(-1, 1, 28, 28), "gan_sampled.png")
