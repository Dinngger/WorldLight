
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms

class TCNN(nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()
        self.cnn = nn.Conv2d(1, 7, 6, 2, 2) # (B, 7, 14, 14)  2 offset + 1 attention + 4 features
        self.pointNet = nn.Sequential(
            nn.Conv1d()
        )

    def forward(self, x):
        x = self.cnn(x)
        return x.squeeze()

mnist_train = torchvision.datasets.MNIST('/media/dinger/inner/Dataset/pytorch_data',
    train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('/media/dinger/inner/Dataset/pytorch_data',
    train=False, download=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")
