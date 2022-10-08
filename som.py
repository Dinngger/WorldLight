# export XLA_PYTHON_CLIENT_PREALLOCATE=false
import os
import numpy as np
import jax.numpy as jnp
from jax import jit
import torch
import torchvision
from torchvision import transforms
import taichi as ti

rows, columns = 10, 5
num = (rows, columns)
shape = (1, 28, 28)
neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

weight = np.random.randn(*num, *shape)
bias = np.zeros(num)

def train(weight, bias, input, alpha):
    x = weight * jnp.expand_dims(input, [0, 1])
    res = jnp.sum(x.reshape(*x.shape[0:2], -1), axis=-1) + bias
    bias = bias - res * 0.01
    i, j = jnp.unravel_index(jnp.argmax(res), res.shape)
    weight = weight.at[i, j].set(jnp.clip(weight[i, j] + jnp.where(input > 0, input, -1) * 0.1, -1, 1))
    # for di, dj in neighbors:
    #     ix, jx = i + di, j + dj
    #     weight = weight.at[ix, jx].set(jnp.clip(weight[ix, jx] + jnp.where(input > 0, input, -1) * alpha, -1, 1))
    return weight, bias, alpha * 0.99

train_jit = jit(train)
mnist = torchvision.datasets.MNIST('/media/dinger/inner/Dataset/pytorch_data', train=True, download=False, transform=transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(
    mnist, batch_size=1)

gui = ti.GUI("SOM", (rows * 28, columns * 28))

def make_grid(data):
    data = ((data + 1) * 128).astype(jnp.uint8)
    data_row = []
    for row in range(rows):
        data_columns = []
        for column in range(columns):
            data_columns.append(data[row, column])
        data_row.append(jnp.hstack(data_columns))
    data = jnp.vstack(data_row)
    return data[:,::-1]

alpha = 0.1
make_grid_jit = jit(make_grid)
while True:
    for x, y in dataloader:
        if not gui.running:
            os._exit(0)
        img = jnp.transpose(weight, [0,1,4,3,2])
        img = make_grid_jit(img)
        gui.set_image(np.asarray(img))
        weight, bias, alpha = train_jit(weight, bias, x.numpy().squeeze(0), alpha)
        gui.show()
